"""Real knowledge base from Wikidata: Nobel laureates as 4 linked tables.

people        person_id, name, birth_date, death_date, gender, birth_place_id, death_place_id
places        place_id, name, country
events        event_id, name, prize, year, place_id            (one Nobel award per prize+year)
relationships src, rel, dst, year                              (person->event 'received', person<->person family)
"""
import os

import pandas as pd

RAW = "bench/wikidata_kb"
REL = {"P26": "spouse", "P40": "child", "P22": "father", "P25": "mother", "P3373": "sibling"}
CEREMONY = {"Nobel Peace Prize": ("Q585", "Oslo", "Norway")}       # every other prize: Stockholm
STOCKHOLM = ("Q1754", "Stockholm", "Sweden")


def qid(u):
    return str(u).rsplit("/", 1)[-1] if isinstance(u, str) and u else None


def date(v):
    return str(v)[:10] if isinstance(v, str) and v[:4].lstrip("-").isdigit() else None


def load():
    ppl = pd.read_csv(f"{RAW}/people.csv", dtype=str, keep_default_na=False)
    plc = pd.read_csv(f"{RAW}/places.csv", dtype=str, keep_default_na=False)
    awd = pd.read_csv(f"{RAW}/awards.csv", dtype=str, keep_default_na=False)
    fam = pd.read_csv(f"{RAW}/family.csv", dtype=str, keep_default_na=False)

    people = pd.DataFrame({
        "person_id": ppl.p.map(qid), "name": ppl.pLabel, "birth_date": ppl.birth.map(date),
        "death_date": ppl.death.map(date), "gender": ppl.genderLabel.replace("", None),
        "birth_place_id": ppl.bp.map(qid), "death_place_id": ppl.dp.map(qid),
    }).drop_duplicates("person_id")                      # Wikidata may hold 2 birth dates: keep first
    people = people[~people.name.str.match(r"^Q\d+$")]  # unlabeled items

    places = pd.DataFrame({"place_id": plc.pl.map(qid), "name": plc.plLabel,
                           "country": plc.countryLabel.replace("", None)})
    places = places[~places.name.str.match(r"^Q\d+$")].drop_duplicates("place_id")
    for pid, nm, ct in [STOCKHOLM, *CEREMONY.values()]:
        if pid not in set(places.place_id):
            places = pd.concat([places, pd.DataFrame([{"place_id": pid, "name": nm, "country": ct}])])

    awd = awd.assign(person=awd.p.map(qid), prize=awd.awardLabel, year=awd.time.str[:4])
    awd = awd[awd.person.isin(people.person_id) & awd.year.str.isdigit()]
    ev = awd[["prize", "year"]].drop_duplicates().sort_values(["year", "prize"]).reset_index(drop=True)
    ev["event_id"] = [f"EV{i:04d}" for i in range(len(ev))]
    ev["name"] = ev.prize + " " + ev.year
    ev["place_id"] = ev.prize.map(lambda p: CEREMONY.get(p, STOCKHOLM)[0])
    events = ev[["event_id", "name", "prize", "year", "place_id"]]

    key = dict(zip(zip(ev.prize, ev.year), ev.event_id))
    rel_award = pd.DataFrame({"src": awd.person, "rel": "received",
                              "dst": [key[(p, y)] for p, y in zip(awd.prize, awd.year)], "year": awd.year})
    fam = pd.DataFrame({"src": fam.a.map(qid), "rel": fam.rel.map(lambda u: REL[qid(u)]),
                        "dst": fam.b.map(qid), "year": None})
    fam = fam[fam.src.isin(people.person_id) & fam.dst.isin(people.person_id)]
    rels = pd.concat([rel_award, fam], ignore_index=True).drop_duplicates(["src", "rel", "dst"])

    # keep referential integrity of the CLEAN kb
    people.loc[~people.birth_place_id.isin(places.place_id), "birth_place_id"] = None
    people.loc[~people.death_place_id.isin(places.place_id), "death_place_id"] = None
    return {"people": people.reset_index(drop=True), "places": places.reset_index(drop=True),
            "events": events.reset_index(drop=True), "relationships": rels.reset_index(drop=True)}


if __name__ == "__main__":
    kb = load()
    os.makedirs(f"{RAW}/clean", exist_ok=True)
    for n, df in kb.items():
        df.to_csv(f"{RAW}/clean/{n}.csv", index=False)
        print(n, df.shape)
    print(kb["relationships"].rel.value_counts().to_dict())
    print(kb["people"].isna().mean().round(3).to_dict())
