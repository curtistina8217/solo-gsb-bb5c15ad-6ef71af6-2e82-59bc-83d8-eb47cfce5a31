"""Tests for the runway slot booking book API (slot-books)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def _book_body(**overrides):
    body = {
        "runway": {
            "runway": "36L",
            "flights": [
                {"flight_no": "CA1", "scheduled_off": "2026-09-18T00:00:00Z", "wake": "J"},
                {"flight_no": "CA2", "scheduled_off": "2026-09-18T00:05:00Z", "wake": "M"},
            ],
        },
    }
    body.update(overrides)
    return body


def _res_body(flight_no="CA9", earliest="2026-09-18T00:03:00Z",
              latest="2026-09-18T00:10:00Z", wake="M"):
    return {
        "flight_no": flight_no,
        "wake": wake,
        "earliest_off": earliest,
        "latest_off": latest,
    }


def _make_book(client, **overrides) -> str:
    r = client.post("/api/v1/slot-books", json=_book_body(**overrides))
    assert r.status_code == 200, r.text
    return r.json()["book_id"]


# ---------- create book ----------


def test_create_book_freezes_plan(client):
    r = client.post("/api/v1/slot-books", json=_book_body())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["book_id"]
    assert data["runway"] == "36L"
    assert data["created_at"]
    assert data["matrix"]["J"]["M"] == 180  # default matrix frozen
    assert [f["flight_no"] for f in data["flights"]] == ["CA1", "CA2"]
    assert data["flights"][0]["scheduled_off"] == "2026-09-18T00:00:00Z"
    assert data["capacity"] is None
    assert data["closures"] == []
    assert data["reservations"] == []
    assert data["active_reservations"] == 0


def test_create_book_freezes_constraints(client):
    body = _book_body()
    body["runway"]["closures"] = [
        {"start": "2026-09-18T00:10:00Z", "end": "2026-09-18T00:20:00Z"}
    ]
    body["runway"]["capacity"] = {"window_seconds": 600, "max_departures": 2}
    body["matrix"] = [{"leader": "J", "follower": "M", "seconds": 90}]
    r = client.post("/api/v1/slot-books", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["matrix"] == {"J": {"M": 90}}  # custom matrix frozen as given
    assert data["closures"] == [
        {"start": "2026-09-18T00:10:00Z", "end": "2026-09-18T00:20:00Z"}
    ]
    assert data["capacity"] == {"window_seconds": 600, "max_departures": 2}


def test_create_book_infeasible_plan_rejected(client):
    # wake violation: J->M needs 180s, plan has 30s
    body = _book_body()
    body["runway"]["flights"][1]["scheduled_off"] = "2026-09-18T00:00:30Z"
    r = client.post("/api/v1/slot-books", json=body)
    assert r.status_code == 422
    assert "尾流间隔违规" in r.json()["detail"]

    # closure conflict
    body = _book_body()
    body["runway"]["closures"] = [
        {"start": "2026-09-18T00:04:00Z", "end": "2026-09-18T00:06:00Z"}
    ]
    r = client.post("/api/v1/slot-books", json=body)
    assert r.status_code == 422
    assert "关闭窗口冲突" in r.json()["detail"]

    # capacity overflow
    body = _book_body()
    body["runway"]["flights"].append(
        {"flight_no": "CA3", "scheduled_off": "2026-09-18T00:06:00Z", "wake": "M"}
    )
    body["runway"]["capacity"] = {"window_seconds": 600, "max_departures": 2}
    body["matrix"] = [{"leader": "J", "follower": "M", "seconds": 0}]
    r = client.post("/api/v1/slot-books", json=body)
    assert r.status_code == 422
    assert "滚动容量超限" in r.json()["detail"]


def test_create_book_infeasible_leaves_no_record(client):
    body = _book_body()
    body["runway"]["flights"][1]["scheduled_off"] = "2026-09-18T00:00:30Z"
    client.post("/api/v1/slot-books", json=body)
    # nothing to look up: the only book id ever returned belongs to a good book
    good = _make_book(client)
    assert client.get(f"/api/v1/slot-books/{good}").status_code == 200


def test_create_book_validation_errors(client):
    body = _book_body()
    body["runway"]["flights"][0]["scheduled_off"] = "2026-09-18T00:00:00"  # naive
    assert client.post("/api/v1/slot-books", json=body).status_code == 422

    body = _book_body()
    body["runway"]["flights"][0]["scheduled_off"] = "2026-09-18T00:00:00.500Z"
    r = client.post("/api/v1/slot-books", json=body)
    assert r.status_code == 422 and "整秒" in r.text

    body = _book_body()
    body["matrix"] = [
        {"leader": "J", "follower": "M", "seconds": 90},
        {"leader": "J", "follower": "M", "seconds": 120},
    ]
    assert client.post("/api/v1/slot-books", json=body).status_code == 422

    body = _book_body()
    body["unexpected"] = 1
    assert client.post("/api/v1/slot-books", json=body).status_code == 422


# ---------- query book ----------


def test_get_book_roundtrip(client):
    book_id = _make_book(client)
    r = client.get(f"/api/v1/slot-books/{book_id}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["book_id"] == book_id
    assert [f["flight_no"] for f in data["flights"]] == ["CA1", "CA2"]


def test_get_unknown_book_returns_404(client):
    r = client.get("/api/v1/slot-books/does-not-exist")
    assert r.status_code == 404
    assert "预约簿不存在" in r.json()["detail"]


# ---------- book reservation ----------


def test_book_reservation_happy_path(client):
    book_id = _make_book(client)
    r = client.post(
        f"/api/v1/slot-books/{book_id}/reservations", json=_res_body()
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["reservation_id"]
    assert data["book_id"] == book_id
    assert data["flight_no"] == "CA9"
    assert data["assigned_off"] == "2026-09-18T00:03:00Z"  # J->M = 180s
    assert data["position"] == 2
    assert data["status"] == "active"
    assert data["cancelled_at"] is None
    # echoed back on the book
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["active_reservations"] == 1
    assert book["reservations"][0]["reservation_id"] == data["reservation_id"]
    assert book["reservations"][0]["assigned_off"] == "2026-09-18T00:03:00Z"


def test_book_reservation_reuses_committed_slots(client):
    """后到请求依据已提交预约重新计算：同一最早时刻被逐次后移。"""
    book_id = _make_book(client)
    r1 = client.post(
        f"/api/v1/slot-books/{book_id}/reservations",
        json=_res_body("CA91", earliest="2026-09-18T00:03:00Z"),
    )
    r2 = client.post(
        f"/api/v1/slot-books/{book_id}/reservations",
        json=_res_body("CA92", earliest="2026-09-18T00:03:00Z"),
    )
    assert r1.status_code == 200 and r2.status_code == 200
    # first books 00:03:00; second must wait M->M = 60s behind it
    assert r1.json()["assigned_off"] == "2026-09-18T00:03:00Z"
    assert r2.json()["assigned_off"] == "2026-09-18T00:04:00Z"


def test_book_reservation_no_slot_returns_409(client):
    book_id = _make_book(client)
    r = client.post(
        f"/api/v1/slot-books/{book_id}/reservations",
        json=_res_body(earliest="2026-09-18T00:00:00Z", latest="2026-09-18T00:02:59Z"),
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "无可行空档" in detail
    assert "2026-09-18T00:03:00Z" in detail  # global earliest reported
    # failure leaves no record
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["reservations"] == []
    assert book["active_reservations"] == 0


def test_book_reservation_duplicate_flight_no_returns_409(client):
    book_id = _make_book(client)
    # duplicate of a frozen-plan flight
    r = client.post(
        f"/api/v1/slot-books/{book_id}/reservations", json=_res_body("CA1")
    )
    assert r.status_code == 409
    assert "重复" in r.json()["detail"]
    # duplicate of an active reservation
    assert (
        client.post(
            f"/api/v1/slot-books/{book_id}/reservations", json=_res_body("CA9")
        ).status_code
        == 200
    )
    r = client.post(
        f"/api/v1/slot-books/{book_id}/reservations", json=_res_body("CA9")
    )
    assert r.status_code == 409
    # still exactly one reservation
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["active_reservations"] == 1


def test_book_reservation_unknown_book_returns_404(client):
    r = client.post(
        "/api/v1/slot-books/does-not-exist/reservations", json=_res_body()
    )
    assert r.status_code == 404


def test_book_reservation_validation_errors(client):
    book_id = _make_book(client)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    # earliest after latest
    r = client.post(url, json=_res_body(latest="2026-09-18T00:02:00Z"))
    assert r.status_code == 422 and "最晚" in r.text
    # naive datetime
    body = _res_body()
    body["earliest_off"] = "2026-09-18T00:03:00"
    assert client.post(url, json=body).status_code == 422
    # sub-second precision
    body = _res_body()
    body["earliest_off"] = "2026-09-18T00:03:00.500Z"
    r = client.post(url, json=body)
    assert r.status_code == 422 and "整秒" in r.text
    # invalid wake / extra field
    body = _res_body()
    body["wake"] = "X"
    assert client.post(url, json=body).status_code == 422
    body = _res_body()
    body["locked"] = True
    assert client.post(url, json=body).status_code == 422
    # nothing persisted after all these failures
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["reservations"] == []


def test_book_reservation_respects_frozen_constraints(client):
    body = _book_body()
    body["runway"]["closures"] = [
        {"start": "2026-09-18T00:10:00Z", "end": "2026-09-18T00:20:00Z"}
    ]
    body["runway"]["capacity"] = {"window_seconds": 600, "max_departures": 2}
    body["matrix"] = [{"leader": "J", "follower": "M", "seconds": 0}]
    book_id = _make_book(client, **body)
    r = client.post(
        f"/api/v1/slot-books/{book_id}/reservations",
        json=_res_body(latest="2026-09-18T00:30:00Z"),
    )
    assert r.status_code == 200, r.text
    # capacity pushes the third departure to 00:10:00, inside the closure
    # window -> pushed to its end 00:20:00 (same as slots/insert would)
    assert r.json()["assigned_off"] == "2026-09-18T00:20:00Z"


# ---------- cancel reservation ----------


def test_cancel_reservation_releases_slot(client):
    book_id = _make_book(client)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    first = client.post(url, json=_res_body("CA91")).json()
    second = client.post(url, json=_res_body("CA92")).json()
    assert first["assigned_off"] == "2026-09-18T00:03:00Z"
    assert second["assigned_off"] == "2026-09-18T00:04:00Z"

    # cancel the first reservation: its 00:03:00 slot is freed immediately
    r = client.delete(
        f"/api/v1/slot-books/{book_id}/reservations/{first['reservation_id']}"
    )
    assert r.status_code == 204
    assert not r.content

    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["active_reservations"] == 1
    by_id = {r["reservation_id"]: r for r in book["reservations"]}
    assert by_id[first["reservation_id"]]["status"] == "cancelled"
    assert by_id[first["reservation_id"]]["cancelled_at"] is not None
    assert by_id[second["reservation_id"]]["status"] == "active"

    # a new booking grabs the released 00:03:00 slot again
    third = client.post(url, json=_res_body("CA93"))
    assert third.status_code == 200
    assert third.json()["assigned_off"] == "2026-09-18T00:03:00Z"


def test_cancel_reservation_is_idempotent(client):
    book_id = _make_book(client)
    reservation = client.post(
        f"/api/v1/slot-books/{book_id}/reservations", json=_res_body()
    ).json()
    url = f"/api/v1/slot-books/{book_id}/reservations/{reservation['reservation_id']}"
    assert client.delete(url).status_code == 204
    assert client.delete(url).status_code == 204  # repeated cancel -> 204


def test_cancel_unknown_returns_404(client):
    book_id = _make_book(client)
    # unknown reservation
    r = client.delete(f"/api/v1/slot-books/{book_id}/reservations/nope")
    assert r.status_code == 404
    assert "预约不存在" in r.json()["detail"]
    # unknown book
    r = client.delete("/api/v1/slot-books/nope/reservations/nope")
    assert r.status_code == 404
    assert "预约簿不存在" in r.json()["detail"]


def test_cancelled_flight_no_is_reusable(client):
    book_id = _make_book(client)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    reservation = client.post(url, json=_res_body("CA9")).json()
    client.delete(f"{url}/{reservation['reservation_id']}")
    r = client.post(url, json=_res_body("CA9"))
    assert r.status_code == 200
    assert r.json()["flight_no"] == "CA9"


def test_frozen_plan_flights_are_not_cancellable(client):
    """固定计划不可取消：冻结航班没有预约 ID，任何非预约 ID 都是 404。"""
    book_id = _make_book(client)
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert all("reservation_id" not in f for f in book["flights"])
    for flight in book["flights"]:
        r = client.delete(
            f"/api/v1/slot-books/{book_id}/reservations/{flight['flight_no']}"
        )
        assert r.status_code == 404
    # the frozen plan is untouched
    again = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert [f["flight_no"] for f in again["flights"]] == ["CA1", "CA2"]


def test_cancel_middle_reservation_wake_gap_returns_409(client):
    """取消中间预约会使新的相邻对违反尾流间隔：拒绝取消并列出冲突明细。

    固定 J 机 00:00，预约 M 机 00:01、L 机 00:02，自定义 J->L=300 秒、
    其他组合为 0：取消 M 后 J->L 只剩 120 秒，不能直接成功。
    """
    body = {
        "runway": {
            "runway": "36L",
            "flights": [
                {"flight_no": "J1", "scheduled_off": "2026-09-18T00:00:00Z", "wake": "J"},
            ],
        },
        "matrix": [{"leader": "J", "follower": "L", "seconds": 300}],
    }
    book_id = _make_book(client, **body)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    m = client.post(
        url,
        json=_res_body(
            "M1",
            earliest="2026-09-18T00:01:00Z",
            latest="2026-09-18T00:01:00Z",
            wake="M",
        ),
    ).json()
    l = client.post(
        url,
        json=_res_body(
            "L1",
            earliest="2026-09-18T00:02:00Z",
            latest="2026-09-18T00:02:00Z",
            wake="L",
        ),
    ).json()
    assert m["assigned_off"] == "2026-09-18T00:01:00Z"
    assert l["assigned_off"] == "2026-09-18T00:02:00Z"

    # cancel the middle M flight: J->L would drop to 120s < 300s -> 409
    r = client.delete(f"{url}/{m['reservation_id']}")
    assert r.status_code == 409
    data = r.json()
    assert "无法取消" in data["detail"]
    assert "J1" in data["detail"] and "L1" in data["detail"]
    assert "300" in data["detail"] and "120" in data["detail"]
    assert len(data["conflicts"]) == 1
    conflict = data["conflicts"][0]
    assert conflict["leader"] == "J1"
    assert conflict["follower"] == "L1"
    assert conflict["leader_wake"] == "J"
    assert conflict["follower_wake"] == "L"
    assert conflict["leader_scheduled_off"] == "2026-09-18T00:00:00Z"
    assert conflict["follower_scheduled_off"] == "2026-09-18T00:02:00Z"
    assert conflict["required_seconds"] == 300
    assert conflict["actual_seconds"] == 120
    assert conflict["shortfall_seconds"] == 180
    assert conflict["leader_position"] == 1
    assert conflict["follower_position"] == 2

    # 预约状态及取消时间均不变，时隙未释放
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    assert book["active_reservations"] == 2
    assert by_id[m["reservation_id"]]["status"] == "active"
    assert by_id[m["reservation_id"]]["cancelled_at"] is None
    assert by_id[l["reservation_id"]]["status"] == "active"

    # 簿状态未被破坏：后续预约不再被 409 拒绝
    after = client.post(url, json=_res_body("M2", earliest="2026-09-18T00:03:00Z",
                                            latest="2026-09-18T00:03:00Z", wake="M"))
    assert after.status_code == 200, after.text

    # 取消 L 机后剩余 J->M 无最小间隔，M 机随之可以取消（其他航班不移动）
    assert client.delete(f"{url}/{l['reservation_id']}").status_code == 204
    assert client.delete(f"{url}/{m['reservation_id']}").status_code == 204
    final = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert final["active_reservations"] == 1  # 之后预约的 M2 仍有效
    assert [f["flight_no"] for f in final["flights"]] == ["J1"]


def test_cancel_middle_reservation_feasible_when_gap_holds(client):
    """取消中间预约后新相邻对仍满足间隔时正常取消，其他航班时刻不变。"""
    # J->L=180 秒，其余组合为 0：J 00:00、M 00:01、L 00:05，取消 M 后
    # J->L=300 秒 >= 180 秒，取消可行。
    body = {
        "runway": {
            "runway": "36L",
            "flights": [
                {"flight_no": "J1", "scheduled_off": "2026-09-18T00:00:00Z", "wake": "J"},
            ],
        },
        "matrix": [{"leader": "J", "follower": "L", "seconds": 180}],
    }
    book_id = _make_book(client, **body)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    m = client.post(
        url,
        json=_res_body(
            "M1",
            earliest="2026-09-18T00:01:00Z",
            latest="2026-09-18T00:01:00Z",
            wake="M",
        ),
    ).json()
    l = client.post(
        url,
        json=_res_body(
            "L1",
            earliest="2026-09-18T00:05:00Z",
            latest="2026-09-18T00:05:00Z",
            wake="L",
        ),
    ).json()

    r = client.delete(f"{url}/{m['reservation_id']}")
    assert r.status_code == 204
    assert not r.content
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    assert by_id[m["reservation_id"]]["status"] == "cancelled"
    assert by_id[m["reservation_id"]]["cancelled_at"] is not None
    # 其他航班时刻不变
    assert by_id[l["reservation_id"]]["assigned_off"] == "2026-09-18T00:05:00Z"
    assert by_id[l["reservation_id"]]["status"] == "active"


# ---------- concurrency ----------


def test_concurrent_bookings_are_serialized(client):
    """跨连接并发预约：读取、求解、写入在同一事务内，后到请求依据已提交
    预约重新计算——全部成功且分配到互不相同的合法时隙。"""
    book_id = _make_book(client)
    url = f"/api/v1/slot-books/{book_id}/reservations"

    def book(i):
        return client.post(
            url,
            json=_res_body(f"CA9{i:02d}", earliest="2026-09-18T00:03:00Z"),
        )

    with ThreadPoolExecutor(max_workers=7) as pool:
        responses = list(pool.map(book, range(7)))

    assert all(r.status_code == 200 for r in responses), [
        (r.status_code, r.text) for r in responses
    ]
    assigned = sorted(r.json()["assigned_off"] for r in responses)
    assert len(set(assigned)) == 7  # no two reservations share a slot
    # 串行结果：00:03:00 起每 60 秒一架（M->M = 60s），跳过被固定计划
    # CA2 占用的 00:05:00
    want = [
        "2026-09-18T00:03:00Z",
        "2026-09-18T00:04:00Z",
        "2026-09-18T00:06:00Z",
        "2026-09-18T00:07:00Z",
        "2026-09-18T00:08:00Z",
        "2026-09-18T00:09:00Z",
        "2026-09-18T00:10:00Z",
    ]
    assert assigned == want


def test_concurrent_bookings_no_slot_leaves_no_record(client):
    """并发下无空档的请求返回 409 且不留记录；成功者互不冲突。"""
    book_id = _make_book(client)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    # window fits exactly two M departures: 00:03:00 and 00:04:00
    def body(i):
        return _res_body(
            f"CA8{i}", earliest="2026-09-18T00:03:00Z", latest="2026-09-18T00:04:00Z"
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda i: client.post(url, json=body(i)), range(6)))

    ok = [r for r in responses if r.status_code == 200]
    conflicted = [r for r in responses if r.status_code == 409]
    assert len(ok) == 2
    assert len(conflicted) == 4
    assert len({r.json()["assigned_off"] for r in ok}) == 2
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["active_reservations"] == 2
    assert len(book["reservations"]) == 2  # failures left nothing behind


def test_concurrent_duplicate_flight_no_exactly_one_wins(client):
    book_id = _make_book(client)
    url = f"/api/v1/slot-books/{book_id}/reservations"

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post(url, json=_res_body("CA9")), range(4)))

    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 409, 409, 409]
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["active_reservations"] == 1


def test_concurrent_cancels_jointly_infeasible_at_most_one_wins(client):
    """两个各自可行但合并不可行的并发取消：至多一个成功（204），另一个 409。

    固定 J@00:00，预约 M@00:01、M@00:02、L@00:03，自定义 J->L=300 秒、
    其他组合为 0：单独取消任一 M 后 J 与 L 之间仍隔着另一 M（可行），
    两个 M 同时取消则 J->L 只剩 180 秒（不可行）。写事务串行化后，
    后到取消依据已提交状态重新校验，必然被 409 拒绝。
    """
    body = {
        "runway": {
            "runway": "36L",
            "flights": [
                {"flight_no": "J1", "scheduled_off": "2026-09-18T00:00:00Z", "wake": "J"},
            ],
        },
        "matrix": [{"leader": "J", "follower": "L", "seconds": 300}],
    }
    book_id = _make_book(client, **body)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    b = client.post(
        url,
        json=_res_body("M1", earliest="2026-09-18T00:01:00Z",
                       latest="2026-09-18T00:01:00Z", wake="M"),
    ).json()
    c = client.post(
        url,
        json=_res_body("M2", earliest="2026-09-18T00:02:00Z",
                       latest="2026-09-18T00:02:00Z", wake="M"),
    ).json()
    d = client.post(
        url,
        json=_res_body("L1", earliest="2026-09-18T00:03:00Z",
                       latest="2026-09-18T00:03:00Z", wake="L"),
    ).json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda rid: client.delete(f"{url}/{rid}"),
                [b["reservation_id"], c["reservation_id"]],
            )
        )

    assert sorted(r.status_code for r in responses) == [204, 409]
    conflicted = next(r for r in responses if r.status_code == 409).json()
    assert conflicted["conflicts"][0]["leader"] == "J1"
    assert conflicted["conflicts"][0]["follower"] == "L1"
    assert conflicted["conflicts"][0]["required_seconds"] == 300
    assert conflicted["conflicts"][0]["actual_seconds"] == 180

    # exactly one of the two middle reservations was cancelled
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    middle_statuses = sorted(
        by_id[r["reservation_id"]]["status"] for r in (b, c)
    )
    assert middle_statuses == ["active", "cancelled"]
    assert by_id[d["reservation_id"]]["status"] == "active"
    # the surviving state is feasible: a fresh booking is not rejected
    again = client.post(
        url,
        json=_res_body("M3", earliest="2026-09-18T00:04:00Z",
                       latest="2026-09-18T00:04:00Z", wake="M"),
    )
    assert again.status_code == 200, again.text


def test_concurrent_cancel_and_booking_share_serialization(client):
    """取消与新预约共用 BEGIN IMMEDIATE 写事务：交错完成后簿状态始终可行。

    固定 J@00:00、已预约 M@00:01 与 L@00:02（J->L=300）。并发执行：
    取消 M 与 在 00:01:30 插入新 M。若取消先提交则其读到 J、M、L 相邻
    而被 409；若插入先提交则取消后仍有新 M 隔开 J、L，取消 204。
    两种交错下预约都成功，且最终固定计划 + 有效预约满足冻结矩阵。
    """
    import threading

    body = {
        "runway": {
            "runway": "36L",
            "flights": [
                {"flight_no": "J1", "scheduled_off": "2026-09-18T00:00:00Z", "wake": "J"},
            ],
        },
        "matrix": [{"leader": "J", "follower": "L", "seconds": 300}],
    }
    book_id = _make_book(client, **body)
    url = f"/api/v1/slot-books/{book_id}/reservations"
    m = client.post(
        url,
        json=_res_body("M1", earliest="2026-09-18T00:01:00Z",
                       latest="2026-09-18T00:01:00Z", wake="M"),
    ).json()
    client.post(
        url,
        json=_res_body("L1", earliest="2026-09-18T00:02:00Z",
                       latest="2026-09-18T00:02:00Z", wake="L"),
    )

    barrier = threading.Barrier(2)

    def cancel():
        barrier.wait()
        return client.delete(f"{url}/{m['reservation_id']}")

    def book():
        barrier.wait()
        return client.post(
            url,
            json=_res_body("M9", earliest="2026-09-18T00:01:30Z",
                           latest="2026-09-18T00:01:30Z", wake="M"),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        cancel_future = pool.submit(cancel)
        book_future = pool.submit(book)
        cancel_r = cancel_future.result()
        book_r = book_future.result()

    assert book_r.status_code == 200, book_r.text
    assert cancel_r.status_code in (204, 409)

    # invariant: the frozen plan plus active reservations still satisfies
    # every frozen adjacent gap (here only J->L = 300s matters)
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    ordered = sorted(
        [(f["flight_no"], f["wake"], _epoch(f["scheduled_off"])) for f in book["flights"]]
        + [
            (r["flight_no"], r["wake"], _epoch(r["assigned_off"]))
            for r in book["reservations"]
            if r["status"] == "active"
        ],
        key=lambda item: (item[2], item[0]),
    )
    required = {("J", "L"): 300}
    for (_, wake_a, t_a), (_, wake_b, t_b) in zip(ordered, ordered[1:]):
        assert t_b - t_a >= required.get((wake_a, wake_b), 0)


def _epoch(iso: str) -> int:
    from datetime import datetime
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
