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


# ---------- cancel with frozen-constraint validation ----------


def _jl_only_book(client):
    """固定计划只有 J@00:00；矩阵只定义 J→L=300 秒，其余组合 0 秒。

    随后预约 M@00:01 与 L@00:02：二者相邻时 J→M、M→L 都无要求，合成
    计划可行；一旦取消中间的 M，J 与 L 直接相邻且只隔 120 秒 < 300 秒。
    """
    body = _book_body()
    body["runway"]["flights"] = [
        {"flight_no": "JX1", "scheduled_off": "2026-09-18T00:00:00Z", "wake": "J"},
    ]
    body["matrix"] = [{"leader": "J", "follower": "L", "seconds": 300}]
    r = client.post("/api/v1/slot-books", json=body)
    assert r.status_code == 200, r.text
    return r.json()["book_id"]


def _book_at(client, book_id, flight_no, wake, when, *, start=None):
    r = client.post(
        f"/api/v1/slot-books/{book_id}/reservations",
        json={
            "flight_no": flight_no,
            "wake": wake,
            "earliest_off": start or when,
            "latest_off": when,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_cancel_middle_reservation_creating_wake_gap_returns_409(client):
    """J(00:00固定) - M(00:01预约) - L(00:02预约)，自定义 J→L=300s：
    取消 M 会让 J 与 L 直接相邻，必须拒绝。"""
    book_id = _jl_only_book(client)
    mid = _book_at(client, book_id, "MM1", "M", "2026-09-18T00:01:00Z")
    assert mid["assigned_off"] == "2026-09-18T00:01:00Z"
    assert mid["position"] == 2
    tail = _book_at(client, book_id, "LY1", "L", "2026-09-18T00:02:00Z")
    assert tail["assigned_off"] == "2026-09-18T00:02:00Z"
    assert tail["position"] == 3

    r = client.delete(
        f"/api/v1/slot-books/{book_id}/reservations/{mid['reservation_id']}"
    )
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["reservation_id"] == mid["reservation_id"]
    assert detail["flight_no"] == "MM1"
    assert detail["assigned_off"] == "2026-09-18T00:01:00Z"
    assert detail["wake_conflict_count"] == 1
    assert detail["conflict_count"] == 1
    conflict = detail["conflicts"][0]
    assert conflict["type"] == "wake"
    assert conflict["leader"] == "JX1"
    assert conflict["follower"] == "LY1"
    assert conflict["leader_wake"] == "J"
    assert conflict["follower_wake"] == "L"
    assert conflict["required_seconds"] == 300
    assert conflict["actual_seconds"] == 120
    assert conflict["shortfall_seconds"] == 180
    # positions are the post-deletion stable order: J then L adjacent
    assert (conflict["leader_position"], conflict["follower_position"]) == (1, 2)
    assert "尾流间隔" in detail["reason"]

    # the reservation stays active, no cancellation timestamp was written
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    assert by_id[mid["reservation_id"]]["status"] == "active"
    assert by_id[mid["reservation_id"]]["cancelled_at"] is None
    assert by_id[tail["reservation_id"]]["status"] == "active"
    assert book["active_reservations"] == 2
    # frozen plan and other flights were not moved
    assert [f["flight_no"] for f in book["flights"]] == ["JX1"]
    assert by_id[tail["reservation_id"]]["assigned_off"] == "2026-09-18T00:02:00Z"


def test_409_cancellation_leaves_slot_occupied_for_later_bookings(client):
    """409 后预约仍参与后续插入求解：其时刻前后仍是受保护的相邻关系。"""
    book_id = _jl_only_book(client)
    mid = _book_at(client, book_id, "MM1", "M", "2026-09-18T00:01:00Z")
    _book_at(client, book_id, "LY1", "L", "2026-09-18T00:02:00Z")
    url = f"/api/v1/slot-books/{book_id}/reservations/{mid['reservation_id']}"
    assert client.delete(url).status_code == 409

    # another M wanting the 00:00-00:01 range cannot take MM1's 00:01:00
    # instant: MM1 is still active and the stable (time, flight_no) order
    # places a same-second candidate after it (later than the requested
    # latest), so the earliest feasible instant is strictly before it.
    r = client.post(
        f"/api/v1/slot-books/{book_id}/reservations",
        json={
            "flight_no": "MM2",
            "wake": "M",
            "earliest_off": "2026-09-18T00:00:00Z",
            "latest_off": "2026-09-18T00:01:00Z",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["assigned_off"] != "2026-09-18T00:01:00Z"
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    assert book["active_reservations"] == 3
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    assert by_id[mid["reservation_id"]]["status"] == "active"


def test_cancel_side_reservations_around_blocked_middle_still_feasible(client):
    """中间 M 不可取消，但取消排在 L 之后、不制造 J→L 相邻对的预约成功。"""
    book_id = _jl_only_book(client)
    mid = _book_at(client, book_id, "MM1", "M", "2026-09-18T00:01:00Z")
    _book_at(client, book_id, "LY1", "L", "2026-09-18T00:02:00Z")
    after_l = _book_at(
        client, book_id, "MM3", "M", "2026-09-18T00:10:00Z",
        start="2026-09-18T00:03:00Z",
    )
    assert after_l["assigned_off"] == "2026-09-18T00:03:00Z"
    # cancelling the middle bridge is rejected ...
    assert (
        client.delete(
            f"/api/v1/slot-books/{book_id}/reservations/{mid['reservation_id']}"
        ).status_code
        == 409
    )
    # ... but cancelling a reservation after L changes nothing for J->L
    r = client.delete(
        f"/api/v1/slot-books/{book_id}/reservations/{after_l['reservation_id']}"
    )
    assert r.status_code == 204
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    assert by_id[mid["reservation_id"]]["status"] == "active"
    assert by_id[after_l["reservation_id"]]["status"] == "cancelled"
    assert by_id[after_l["reservation_id"]]["cancelled_at"] is not None
    assert book["active_reservations"] == 2


def test_two_individually_feasible_concurrent_cancels_at_most_one_wins(client):
    """J(00:00固定) - A(预约) - B(预约) - L(00:02预约)，J→L=300s：单独
    取消 A 或 B 都可行（另一架仍夹在 J、L 之间），同时取消则 J 与 L
    直接相邻而不可行；写事务串行化保证至多一个成功。"""
    book_id = _jl_only_book(client)
    first = _book_at(client, book_id, "MA1", "M", "2026-09-18T00:00:01Z")
    second = _book_at(
        client, book_id, "MB2", "M", "2026-09-18T00:02:00Z",
        start="2026-09-18T00:00:03Z",
    )
    assert first["assigned_off"] == "2026-09-18T00:00:01Z"
    assert second["assigned_off"] == "2026-09-18T00:00:03Z"
    tail = _book_at(client, book_id, "LY1", "L", "2026-09-18T00:02:00Z")
    assert tail["assigned_off"] == "2026-09-18T00:02:00Z"

    def cancel(reservation_id):
        return client.delete(
            f"/api/v1/slot-books/{book_id}/reservations/{reservation_id}"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                cancel,
                [first["reservation_id"], second["reservation_id"]],
            )
        )

    codes = sorted(r.status_code for r in responses)
    assert codes == [204, 409], [(r.status_code, r.text) for r in responses]
    loser = next(r for r in responses if r.status_code == 409).json()["detail"]
    assert loser["wake_conflict_count"] == 1
    conflict = loser["conflicts"][0]
    assert (conflict["leader"], conflict["follower"]) == ("JX1", "LY1")
    assert conflict["required_seconds"] == 300
    assert conflict["actual_seconds"] == 120

    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    statuses = {by_id[first["reservation_id"]]["status"],
                by_id[second["reservation_id"]]["status"]}
    assert statuses == {"active", "cancelled"}
    assert by_id[tail["reservation_id"]]["status"] == "active"
    assert book["active_reservations"] == 2  # one bridge + LY1
    # the loser must be retryable after the other bridge comes back ... simply
    # confirm the book still holds a feasible schedule and no slot moved
    assert by_id[tail["reservation_id"]]["assigned_off"] == "2026-09-18T00:02:00Z"


def test_concurrent_duplicate_cancel_both_return_204(client):
    """对同一预约并发重复取消：幂等，二者都 204（仅一次真正取消）。"""
    book_id = _jl_only_book(client)
    _book_at(client, book_id, "LY1", "L", "2026-09-18T00:05:00Z",
             start="2026-09-18T00:03:00Z")
    reservation = _book_at(
        client, book_id, "MM9", "M", "2026-09-18T00:10:00Z",
        start="2026-09-18T00:06:00Z",
    )
    url = f"/api/v1/slot-books/{book_id}/reservations/{reservation['reservation_id']}"
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.delete(url), range(2)))
    assert sorted(r.status_code for r in responses) == [204, 204]
    book = client.get(f"/api/v1/slot-books/{book_id}").json()
    by_id = {x["reservation_id"]: x for x in book["reservations"]}
    rec = by_id[reservation["reservation_id"]]
    assert rec["status"] == "cancelled"
    assert rec["cancelled_at"] is not None


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
