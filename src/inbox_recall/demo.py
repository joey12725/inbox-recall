"""A fabricated, multi-network message archive for demos, tests and the benchmark.

Everyone here is made up. The archive belongs to "Jordan Lee" and spans
WhatsApp, Signal, Telegram, iMessage, Discord and Instagram. Storylines are
anchored to the current date (via the same time parser agents use), so
"last spring" in a query always lands on the trip that happened last spring.
Thousands of filler messages surround them so ranking actually has to work.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import psycopg

from beeper_pg_sync.model import Account, Batch, Conversation, Message, Participant
from beeper_pg_sync.store import Store

from .timeparse import parse_when

SOURCE = "demo"
ME = Participant(native_id="me", display_name="Jordan Lee", handle="@jordan.lee.demo", is_self=True)


def person(native_id: str, name: str, handle: str | None = None) -> Participant:
    return Participant(native_id=native_id, display_name=name, handle=handle)


MAYA_WA = person("15550100201", "Maya Chen")
MAYA_IG = person("maya.makes.demo", "Maya Chen", "@maya.makes.demo")
PRIYA = person("15550100202", "Priya Nair")
THEO = person("theo_park_demo", "Theo Park", "@theopark")
SAM = person("15550100204", "Sam Rivera")
LENA = person("lena#0001", "Lena Ortiz", "@lenao")
KAI = person("kai.moreno.demo", "Kai Moreno", "@kai.moreno.demo")
OMAR = person("15550100207", "Omar Haddad")
MOM = person("15550100208", "Grace Lee")
DAD = person("15550100209", "Daniel Lee")


@dataclass
class Chat:
    key: str
    network: str
    title: str
    kind: str
    members: list[Participant]
    lines: list[tuple[str, datetime, Participant, str]] = field(default_factory=list)

    def say(self, key: str, at: datetime, who: Participant, text: str) -> None:
        self.lines.append((key, at, who, text))


def _script(now: datetime) -> list[Chat]:
    def anchor(phrase: str, days: float = 0, hour: int = 12, minute: int = 0) -> datetime:
        start = parse_when(phrase, now).start
        assert start is not None
        return start.replace(hour=hour, minute=minute) + timedelta(days=days)

    def t(base: datetime, minutes: float) -> datetime:
        return base + timedelta(minutes=minutes)

    maya = Chat("wa-maya", "whatsapp", "Maya Chen", "dm", [MAYA_WA])
    spring = anchor("last spring", 24, 19, 2)
    maya.say(
        "lisbon-1", spring, MAYA_WA, "ok I'm officially in for Lisbon in May!! flights are cheap from here right now"
    )
    maya.say("lisbon-2", t(spring, 3), ME, "amazing. I can do May 14 to 20, does that work?")
    maya.say("lisbon-3", t(spring, 5), MAYA_WA, "perfect. I'll book the flights tonight, TAP has a direct one")
    maya.say("lisbon-4", t(spring, 41), MAYA_WA, "booked! confirmation code is in your email. we land 7:50am")
    stay = t(spring, 60 * 24 * 3)
    maya.say(
        "lisbon-5",
        stay,
        MAYA_WA,
        "found a place in Alfama, the one with the blue tiled balcony and the view of the river",
    )
    maya.say("lisbon-6", t(stay, 2), ME, "omg yes book it before someone else does")
    maya.say("lisbon-7", t(stay, 9), MAYA_WA, "done, 6 nights. you owe me 410 for your half")
    maya.say("lisbon-8", t(stay, 12), ME, "sending it now")
    tips = t(spring, 60 * 24 * 30)
    maya.say(
        "lisbon-9", tips, MAYA_WA, "my coworker says Manteigaria has the best pastel de nata, way better than Belem"
    )
    maya.say(
        "lisbon-10",
        t(tips, 4),
        MAYA_WA,
        "also we have to do the day trip to Sintra, she said go early before the tour buses",
    )
    maya.say("lisbon-11", t(tips, 7), ME, "adding both to the list")
    after = anchor("last spring", 80, 9, 30)
    maya.say("lisbon-12", after, MAYA_WA, "still thinking about that custard tart place. best trip we've taken")

    maya_ig = Chat("ig-maya", "instagram", "maya.makes.demo", "dm", [MAYA_IG])
    clay = anchor("last fall", 18, 21, 15)
    maya_ig.say("pottery-1", clay, MAYA_IG, "this wheel throwing class at Clay Studio looks so fun, saturdays at 10")
    maya_ig.say("pottery-2", t(clay, 6), ME, "I'm in. is it beginner friendly?")
    maya_ig.say("pottery-3", t(clay, 8), MAYA_IG, "yes, 4 week intro course, $180 and they fire everything for you")
    maya_ig.say("pottery-4", anchor("last fall", 40, 13), MAYA_IG, "my bowl came out lopsided but I love it")

    priya = Chat("sig-priya", "signal", "Priya Nair", "dm", [PRIYA])
    lease = anchor("2 months ago", 3, 18, 40)
    priya.say("lease-1", lease, PRIYA, "got an email from Mr. Alvarez, rent goes up to 2,150 when we renew")
    priya.say("lease-2", t(lease, 2), ME, "ugh that's almost 9% more")
    priya.say("lease-3", t(lease, 5), PRIYA, "renewal paperwork is due by the 30th or we go month to month")
    priya.say("lease-4", t(lease, 7), PRIYA, "I think we should sign, moving would cost more than the difference")
    priya.say("lease-5", t(lease, 15), ME, "agreed, let's sign the 12 month one")
    leak = anchor("5 months ago", 8, 8, 5)
    priya.say("leak-1", leak, PRIYA, "the kitchen sink is leaking again, water all over the cabinet")
    priya.say("leak-2", t(leak, 20), PRIYA, "plumber can come thursday between 1 and 4")

    book = Chat("sig-book", "signal", "Book Club", "group", [PRIYA, OMAR, LENA])
    club = anchor("last month", 6, 20, 0)
    book.say("club-1", club, OMAR, "next pick: The Glass Orchard. it's short, we can do it in three weeks")
    book.say("club-2", t(club, 4), LENA, "love it. hosting at my place this time?")
    book.say("club-3", t(club, 9), OMAR, "let's do my place on Thursday the 21st, I'll make the lentil soup")
    book.say("club-4", t(club, 11), ME, "I'll bring bread and the cheap wine")
    book.say("club-5", t(club, 60 * 24 * 12), PRIYA, "halfway through and the ending of chapter 9 wrecked me")

    theo = Chat("tg-theo", "telegram", "Theo Park", "dm", [THEO])
    rec = anchor("last year", 140, 22, 10)
    theo.say(
        "rec-1",
        rec,
        THEO,
        "you HAVE to read Saltwater Archive. sci-fi about a lighthouse keeper who hears signals from the future",
    )
    theo.say("rec-2", t(rec, 3), ME, "adding it. is it a series?")
    theo.say("rec-3", t(rec, 5), THEO, "trilogy, but the first one stands alone")
    cousin = anchor("2 years ago", 12, 17)
    theo.say("cousin-1", cousin, THEO, "my cousin just moved to Lisbon for work, says rent is wild there now")
    dentist = anchor("last month", 2, 11, 30)
    theo.say("dentist-1", dentist, THEO, "Dr. Okafor on Elm Street is the dentist I told you about, super gentle")

    sam = Chat("im-sam", "imessage", "Sam Rivera", "dm", [SAM])
    car = anchor("past 2 weeks", 3, 16, 20)
    sam.say("car-1", car, SAM, "mechanic called, it's the timing belt. 640 with labor")
    sam.say("car-2", t(car, 4), ME, "oof. at least it's not the transmission")
    sam.say("car-3", t(car, 6), SAM, "car will be ready friday afternoon")
    gift = anchor("last month", 20, 19)
    sam.say("gift-1", gift, ME, "what should I get my mom for her birthday? she has everything")
    sam.say(
        "gift-2",
        t(gift, 3),
        SAM,
        "my mom loved the ceramic planter I got her, the speckled one from the farmers market",
    )

    lena = Chat("dc-lena", "discord", "side-project", "group", [LENA, THEO])
    disk = anchor("past 3 weeks", 2, 23, 5)
    lena.say("disk-1", disk, LENA, "staging is down again, the server ran out of disk space")
    lena.say("disk-2", t(disk, 3), THEO, "it's the log files, nobody set up rotation")
    lena.say("disk-3", t(disk, 7), ME, "I'll add logrotate tonight and bump the volume to 50GB")
    launch = anchor("4 months ago", 12, 15)
    lena.say("launch-1", launch, LENA, "beta invite list is at 340 people, we should launch before the holidays")

    kai = Chat("ig-kai", "instagram", "kai.moreno.demo", "dm", [KAI])
    show = anchor("last summer", 30, 14, 45)
    kai.say("show-1", show, KAI, "The Paper Lanterns are playing Lantern Hall on August 2nd, want to go?")
    kai.say("show-2", t(show, 2), ME, "YES. grab two tickets and I'll pay you back")
    kai.say("show-3", t(show, 10), KAI, "got them, 85 each, section B")
    kai.say("show-4", t(show, 14), ME, "sent you 85, thank you!!")
    flight = anchor("past 3 weeks", 9, 9)
    kai.say("flight-1", flight, KAI, "flight to Denver delayed 3 hours, airport ramen it is")

    family = Chat("wa-family", "whatsapp", "Lee Family", "group", [MOM, DAD])
    tg = anchor("last thanksgiving", -6, 10)
    family.say("tg-1", tg, MOM, "Thanksgiving at our house this year. I'm making the sweet potato casserole")
    family.say("tg-2", t(tg, 30), DAD, "my flight lands Wednesday at 3:40pm, can someone pick me up?")
    family.say("tg-3", t(tg, 34), ME, "I'll get you from the airport dad")
    family.say("tg-4", t(tg, 60), MOM, "Jordan please bring the folding chairs from your garage")
    bday = anchor("last month", 25, 9)
    family.say("bday-1", bday, ME, "happy birthday mom!! dinner on saturday is on me")
    family.say("bday-2", t(bday, 15), MOM, "thank you sweetheart, the planter is beautiful")

    return [maya, maya_ig, priya, book, theo, sam, lena, kai, family]


FILLER = [
    "lol", "haha yes", "omw", "sounds good", "ok perfect", "running 10 min late", "good morning!",
    "call me when you can", "did you see the game last night", "can you send me that link",
    "what time works for you", "happy friday", "on my way", "just got home", "that's hilarious",
    "can't wait", "thank you!!", "no worries", "let me check and get back to you", "sure thing",
    "are we still on for tomorrow", "I'm so tired today", "send pics", "same", "wait really?",
    "this weather is unreal", "grabbing coffee, want anything", "see you soon", "love that",
    "sorry just saw this", "yep", "nope", "maybe next week?", "traffic is terrible",
    "did you eat yet", "I'll be there in 5", "that sounds amazing", "ok", "good night",
    "how was your weekend", "work is chaos this week", "let's catch up soon", "hahaha",
    "I forgot my charger again", "the new episode was so good", "noted", "totally",
    "can we push to 7?", "I'm outside", "brb", "true", "fair enough", "congrats!!",
]  # fmt: skip


def build(now: datetime, filler: int = 3000, seed: int = 7) -> list[Chat]:
    """The scripted chats plus deterministic filler spread over the last three years."""
    chats = _script(now)
    rng = random.Random(seed)
    busy = {(c.key, at.date()) for c in chats for _, at, _, _ in c.lines}
    start = now - timedelta(days=3 * 365)
    for n in range(filler):
        chat = chats[rng.randrange(len(chats))]
        for _ in range(20):
            at = start + timedelta(seconds=rng.randrange(int((now - start).total_seconds()) - 3600))
            if (chat.key, at.date()) not in busy:
                break
        who = ME if rng.random() < 0.4 else chat.members[rng.randrange(len(chat.members))]
        chat.say(f"filler-{n}", at, who, rng.choice(FILLER))
    for chat in chats:
        chat.lines.sort(key=lambda line: line[1])
    return chats


def seed(conn: psycopg.Connection, now: datetime, filler: int = 3000, seed: int = 7) -> dict[str, int]:
    """Write the demo archive through the normal ingest path. Returns message counts per network."""
    store = Store(conn, notify=False)
    counts: dict[str, int] = {}
    for chat in build(now, filler, seed):
        account = Account(source_account_id=f"demo-{chat.network}", network=chat.network, user_native_id="me")
        messages = [
            Message(
                source_message_id=f"{chat.key}:{key}",
                body=text,
                sent_at=at,
                direction="outgoing" if who is ME else "incoming",
                sender=who,
            )
            for key, at, who, text in chat.lines
        ]
        batch = Batch(
            source=SOURCE,
            conversation=Conversation(
                source_conversation_id=chat.key, network=chat.network, title=chat.title, kind=chat.kind
            ),
            messages=messages,
            account=account,
            participants=[ME, *chat.members],
        )
        store.ingest(batch)
        counts[chat.network] = counts.get(chat.network, 0) + len(messages)
    return counts


def message_ids(conn: psycopg.Connection, keys: list[str]) -> dict[str, int]:
    """Map demo keys like 'wa-maya:lisbon-5' to message ids."""
    rows = conn.execute(
        "SELECT source_message_id, message_id FROM message_sources WHERE source = %s AND source_message_id = ANY(%s)",
        (SOURCE, keys),
    ).fetchall()
    return {r["source_message_id"]: r["message_id"] for r in rows}
