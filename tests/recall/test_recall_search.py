"""Hybrid retrieval, filters, context windows and pagination on the fabricated archive."""

import pytest
from conftest import NOW, TZ

from inbox_recall.query import SearchRequest
from inbox_recall.search import Searcher


def hit_ids(result):
    return [h["id"] for h in result["hits"]]


def test_hybrid_search_with_person_and_plain_english_time(searcher, ids):
    result = searcher.search(SearchRequest(query="lisbon flights", sender=["maya"], when="last spring", limit=5))
    assert result["mode"] == "hybrid"
    assert set(ids("wa-maya:lisbon-1", "wa-maya:lisbon-3", "wa-maya:lisbon-4")) <= set(hit_ids(result)[:5])
    assert result["filters"]["time"] == '2026-03-01 to 2026-05-31 ("last spring")'
    for excerpt in result["excerpts"]:
        for m in excerpt["messages"]:
            if "hit" in m:
                assert m["sender"] == "Maya Chen" and m["sent_at"].startswith("2026-0")


def test_hits_carry_citations(searcher, ids):
    result = searcher.search(SearchRequest(query="Manteigaria", limit=1, context=0))
    (message,) = result["excerpts"][0]["messages"]
    assert message["id"] == ids("wa-maya:lisbon-9")[0]
    assert message["chat"] == "Maya Chen" and message["network"] == "whatsapp"
    assert message["sender"] == "Maya Chen" and message["direction"] == "incoming"
    assert message["sent_at"].endswith("-05:00")  # rendered in the configured time zone
    assert "keyword" in result["hits"][0]["matched_by"] and "semantic" in result["hits"][0]["matched_by"]


def test_filters_narrow_across_networks(searcher, ids):
    theo = searcher.search(SearchRequest(query="lisbon", person=["Theo"], limit=3))
    assert hit_ids(theo)[0] == ids("tg-theo:cousin-1")[0]
    # Theo's DM (Telegram) and the side-project group he is in (Discord), nothing else
    assert {e["conversation"]["network"] for e in theo["excerpts"]} <= {"telegram", "discord"}

    mine = searcher.search(SearchRequest(query="pay you back", sender=["me"], limit=3))
    assert hit_ids(mine)[0] == ids("ig-kai:show-2")[0]

    family = searcher.search(SearchRequest(query="flight", chat=["lee family"], limit=3))
    assert hit_ids(family)[0] == ids("wa-family:tg-2")[0]

    insta = searcher.search(SearchRequest(query="tickets", network=["instagram"], when="last summer", limit=5))
    assert set(ids("ig-kai:show-1", "ig-kai:show-3")) <= set(hit_ids(insta))

    incoming = searcher.search(SearchRequest(query="lisbon", direction="outgoing", limit=50, context=0))
    assert all(m["direction"] == "outgoing" for e in incoming["excerpts"] for m in e["messages"])


def test_fuzzy_matching_rescues_typos(searcher, ids):
    assert hit_ids(searcher.search(SearchRequest(query="manteigria", limit=3)))[0] == ids("wa-maya:lisbon-9")[0]
    keyword_only = searcher.search(SearchRequest(query="Lantern Hal", mode="keyword", limit=3))
    assert ids("ig-kai:show-1")[0] in hit_ids(keyword_only)


def test_context_windows_merge_into_one_excerpt_per_thread(searcher, ids):
    result = searcher.search(SearchRequest(query="lisbon flights", sender=["maya"], when="last spring", context=2))
    first = result["excerpts"][0]
    assert len(first["hit_ids"]) >= 2  # neighbouring hits share one excerpt
    message_ids = [m["id"] for m in first["messages"]]
    assert message_ids == sorted(message_ids, key=lambda i: message_ids.index(i))
    times = [m["sent_at"] for m in first["messages"]]
    assert times == sorted(times)
    lisbon_2 = ids("wa-maya:lisbon-2")[0]
    assert lisbon_2 in message_ids  # context between two hits
    assert all("hit" not in m for m in first["messages"] if m["id"] == lisbon_2)


def test_reply_targets_are_quoted(archive, searcher, ids):
    planter, birthday = ids("wa-family:bday-2", "wa-family:bday-1")
    archive.execute("UPDATE messages SET reply_to_message_id = %s WHERE id = %s", (birthday, planter))
    result = searcher.search(SearchRequest(query="the planter is beautiful", chat=["Lee Family"], limit=1, context=0))
    (message,) = result["excerpts"][0]["messages"]
    assert message["in_reply_to"]["id"] == birthday
    assert message["in_reply_to"]["text"].startswith("happy birthday mom")


def test_relevance_pagination_is_stable_and_has_no_duplicates(searcher):
    req = SearchRequest(query="good morning", limit=7, context=0)
    seen = []
    page = searcher.search(req)
    for _ in range(4):
        seen.extend(hit_ids(page))
        if not page["next_cursor"]:
            break
        page = searcher.search(SearchRequest(query="good morning", limit=7, context=0, cursor=page["next_cursor"]))
    assert len(seen) == len(set(seen)) == 28
    one_shot = searcher.search(SearchRequest(query="good morning", limit=28, context=0))
    assert hit_ids(one_shot) == seen

    with pytest.raises(ValueError, match="different search"):
        searcher.search(SearchRequest(query="something else", cursor=page["next_cursor"]))


def test_empty_query_browses_every_match_newest_first(searcher, archive):
    total = archive.execute(
        """SELECT count(*) AS n FROM messages m JOIN conversations c ON c.id = m.conversation_id
           WHERE c.title = 'Maya Chen' AND m.body <> '' AND NOT m.deleted"""
    ).fetchone()["n"]
    seen, cursor = [], None
    while True:
        page = searcher.search(SearchRequest(chat=["Maya Chen"], limit=25, context=0, cursor=cursor))
        assert page["mode"] == "browse"
        seen.extend(hit_ids(page))
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(seen) == len(set(seen)) == total
    stamps = [archive.execute("SELECT sent_at FROM messages WHERE id = %s", (i,)).fetchone()["sent_at"] for i in seen]
    assert stamps == sorted(stamps, reverse=True)


def test_without_embeddings_search_falls_back_to_keywords(archive):
    plain = Searcher(archive, None, tz=TZ, now=NOW)
    result = plain.search(SearchRequest(query="timing belt"))
    assert result["mode"] == "keyword" and result["notes"]
    assert "keyword" in result["hits"][0]["matched_by"] and "semantic" not in result["hits"][0]["matched_by"]


def test_semantic_only_mode(searcher, ids):
    result = searcher.search(SearchRequest(query="sweet potato casserole", mode="semantic", limit=3))
    assert result["mode"] == "semantic"
    assert hit_ids(result)[0] == ids("wa-family:tg-1")[0]
    assert all(h["matched_by"] == ["semantic"] for h in result["hits"])


def test_get_conversation_reconstructs_threads_with_cursors(searcher, ids):
    center = ids("wa-maya:lisbon-3")[0]
    thread = searcher.conversation(message_id=center, before=2, after=1)
    got = [m["id"] for m in thread["messages"]]
    assert got == [*ids("wa-maya:lisbon-1", "wa-maya:lisbon-2"), center, ids("wa-maya:lisbon-4")[0]]
    assert [m for m in thread["messages"] if m.get("hit")][0]["id"] == center

    newer = searcher.conversation(cursor=thread["newer_cursor"], limit=2)
    assert [m["id"] for m in newer["messages"]] == ids("wa-maya:lisbon-5", "wa-maya:lisbon-6")
    older = searcher.conversation(cursor=thread["older_cursor"], limit=3)
    assert len(older["messages"]) == 3 and older["messages"][-1]["sent_at"] < thread["messages"][0]["sent_at"]

    by_time = searcher.conversation(conversation_id=thread["conversation"]["id"], when="last spring", limit=200)
    assert all(m["sent_at"][:7] in ("2026-03", "2026-04", "2026-05") for m in by_time["messages"])
    assert center in [m["id"] for m in by_time["messages"]]

    with pytest.raises(ValueError):
        searcher.conversation()


def test_people_merge_across_networks(searcher):
    people = searcher.people("maya")["people"]
    assert len(people) == 1
    maya = people[0]
    assert maya["name"] == "Maya Chen"
    assert set(maya["networks"]) == {"whatsapp", "instagram"}
    assert "@maya.makes.demo" in maya["handles"]
    top = searcher.people()["people"]
    assert all(p["name"] != "Jordan Lee" for p in top)  # the archive owner is not a contact


def test_list_conversations_with_filters(searcher):
    chats = searcher.conversations(person=["Priya"])["conversations"]
    assert {c["title"] for c in chats} == {"Priya Nair", "Book Club"}
    book = next(c for c in chats if c["title"] == "Book Club")
    assert set(book["people"]) == {"Priya Nair", "Omar Haddad", "Lena Ortiz"}
    assert searcher.conversations(network=["discord"])["conversations"][0]["title"] == "side-project"
    paged = searcher.conversations(limit=4)
    rest = searcher.conversations(limit=4, cursor=paged["next_cursor"])
    assert len(paged["conversations"]) == 4 and rest["conversations"][0] not in paged["conversations"]


def test_stats_report_coverage(searcher):
    stats = searcher.stats()
    assert {n["network"] for n in stats["networks"]} == {
        "whatsapp",
        "instagram",
        "signal",
        "telegram",
        "imessage",
        "discord",
    }
    (model,) = stats["embedding_models"]
    assert model["active"] and model["coverage"] == 1.0
