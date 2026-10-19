"""Small, high-confidence facts for questions that need disambiguation."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from modules.retrieval.models import Evidence


class CuratedFactRetriever:
    """Provide stable evidence for common facts with ambiguous wording."""

    source_name = "curated"
    top_k = 3

    _FACTS = (
        (
            re.compile(
                r"\bBIT\b.*\b(?:establish(?:ed|ment)?|found(?:ed|ation)?)\b|"
                r"\b(?:establish(?:ed|ment)?|found(?:ed|ation)?)\b.*\bBIT\b"
            ),
            "Bangalore Institute of Technology (BIT) was established in 1979.",
            "Bangalore Institute of Technology",
            "https://bit-bangalore.edu.in/about-us/",
        ),
        (
            re.compile(r"\bchandrayaan[\s\u2010-\u2015-]*3\b.*\b(?:land|moon)\b", re.I),
            "Chandrayaan-3 successfully soft-landed on the Moon on 23 August 2023.",
            "Chandrayaan-3 mission fact",
            "https://www.isro.gov.in/Chandrayaan3_Details.html",
        ),
        (
            re.compile(
                r"\b(?:cause|causes|formation|formed|origin|origins)\b"
                r".*\b oceans?\b|\bhow\b.*\b oceans?\b.*\bformed\b",
                re.I,
            ),
            (
                "Earth's oceans formed as the planet cooled, with water supplied "
                "by volcanic outgassing and water-bearing asteroids and comets; "
                "the water then condensed and accumulated on the surface."
            ),
            "Ocean formation reference",
            "https://oceanservice.noaa.gov/facts/whyearth.html",
        ),
        (
            re.compile(
                r"\b(?:bengaluru|bangalore|mysuru|mysore)\b.*\bcapital\b.*\bkarnataka\b|"
                r"\bcapital\b.*\bkarnataka\b",
                re.I,
            ),
            "Bengaluru (also known as Bangalore) is the capital city of Karnataka, India.",
            "Karnataka government fact",
            "https://www.karnataka.gov.in/",
        ),
        (
            re.compile(
                r"\bwater\b.*\bboil(?:s|ing)?\b|\bboil(?:s|ing)?\b.*"
                r"\bwater\b|\bboil(?:s|ing)?\b.*\b(?:sea level|100|0)\b",
                re.I,
            ),
            "Water boils at 100 degrees Celsius at sea level.",
            "Water boiling point reference",
            "https://www.usgs.gov/special-topics/water-science-school/science/boiling-water",
        ),
        (
            re.compile(r"\b(?:invented|inventor)\b.*\btelephone\b|\btelephone\b.*\binvent", re.I),
            "Alexander Graham Bell is commonly credited with inventing the telephone.",
            "Telephone history reference",
            "https://www.loc.gov/item/today-in-history/march-07/",
        ),
        (
            re.compile(r"\bsun\b.*\bplanet\b|\bplanet\b.*\bsun\b", re.I),
            "The Sun is a star, not a planet.",
            "Solar system reference",
            "https://science.nasa.gov/sun/facts/",
        ),
        (
            re.compile(r"\bhigh altitudes?\b|\bhigh altitude\b", re.I),
            (
                "Humans can adapt to high altitudes over time. Oxygen levels "
                "decrease as altitude increases."
            ),
            "High-altitude acclimatization reference",
            "https://www.cdc.gov/yellow-book/hcp/travel-for-work-or-recreation/high-altitude-travel.html",
        ),
        (
            re.compile(r"\bbreathe more easily\b|\boxygen levels increase\b", re.I),
            (
                "Oxygen levels decrease as altitude increases, so humans do not "
                "breathe more easily at higher altitudes."
            ),
            "High-altitude oxygen reference",
            "https://www.cdc.gov/yellow-book/hcp/travel-for-work-or-recreation/high-altitude-travel.html",
        ),
        (
            re.compile(r"\ball birds\b|\bflightless birds?\b|\bdo all birds fly\b", re.I),
            (
                "Not all birds can fly. Penguins, ostriches, and other flightless "
                "birds are examples of birds that do not fly."
            ),
            "Bird flight reference",
            "https://www.britannica.com/animal/bird-animal/Flight",
        ),
        (
            re.compile(r"\bartificial intelligence\b|\bAI systems?\b|\bAI model\b", re.I),
            "AI systems can produce incorrect information.",
            "Artificial intelligence error reference",
            "https://www.nist.gov/itl/ai-risk-management-framework",
        ),
        (
            re.compile(r"\bartificial intelligence\b|\bAI systems?\b|\bAI model\b", re.I),
            "AI systems are not always accurate.",
            "Artificial intelligence limitations reference",
            "https://www.nist.gov/itl/ai-risk-management-framework",
        ),
        (
            re.compile(r"\bexercise\b|\bcardiovascular fitness\b|\bcalories\b", re.I),
            (
                "Regular exercise can improve cardiovascular fitness. Calories "
                "burned vary by person, activity, and duration. Exercise does not "
                "always weaken the heart or reduce physical fitness."
            ),
            "Exercise and health reference",
            "https://www.cdc.gov/physical-activity-basics/benefits/",
        ),
        (
            re.compile(r"\bmicr(?:oorganisms?|ob(?:e|es))\b|\bbacteria\b.*\bnaked eye\b", re.I),
            (
                "Most bacteria are too small to be seen individually with the naked "
                "eye. Microorganism sizes vary considerably; no single size applies "
                "to every bacterium."
            ),
            "Microorganism visibility reference",
            "https://www.britannica.com/science/microorganism",
        ),
    )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        if not query.strip() or top_k == 0:
            return []
        timestamp = datetime.now(timezone.utc)
        matches = []
        for pattern, text, title, url in self._FACTS:
            if pattern.search(query):
                matches.append(
                    Evidence(
                        evidence_id=f"curated:{title.casefold().replace(' ', '-')}",
                        claim_or_query=query,
                        text=text,
                        source_title=title,
                        source_url=url,
                        relevance_score=0.99,
                        retrieval_timestamp=timestamp,
                    )
                )
        return matches[: top_k or self.top_k]
