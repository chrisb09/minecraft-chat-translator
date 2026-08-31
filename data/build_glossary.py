#!/usr/bin/env python3
"""
build_glossary.py

Parses rawdata/minecraft_glossary.txt (list of English Minecraft blocks, items, entities)
and builds a bilingual EN <-> DE domain nomenclature glossary (data/minecraft_bilingual_glossary.json).
"""

import os
import json
import re
from typing import Dict, List

INPUT_GLOSSARY = "/git/finetune/rawdata/minecraft_glossary.txt"
OUTPUT_GLOSSARY = "/git/finetune/data/minecraft_bilingual_glossary.json"

# Core translations for common Minecraft terminology
EN_TO_DE_BASE = {
    # Materials / Ores
    "Ancient Debris": "Antiker Schrott",
    "Netherite Ingot": "Netheritbarren",
    "Netherite Scrap": "Netheritschrott",
    "Netherite Block": "Netheritblock",
    "Netherite Sword": "Netheritschwert",
    "Netherite Pickaxe": "Netheritspitzhacke",
    "Netherite Axe": "Netheritaxt",
    "Netherite Shovel": "Netheritschaufel",
    "Netherite Hoe": "Netherithacke",
    "Netherite Helmet": "Netherithelm",
    "Netherite Chestplate": "Netheritharnisch",
    "Netherite Leggings": "Netheritbeinschutz",
    "Netherite Boots": "Netheritstiefel",
    "Diamond": "Diamant",
    "Diamond Ore": "Diamanterz",
    "Diamond Block": "Diamantblock",
    "Emerald": "Smaragd",
    "Gold Ingot": "Goldbarren",
    "Iron Ingot": "Eisenbarren",
    "Copper Ingot": "Kupferbarren",
    "Redstone": "Redstone",
    "Lapis Lazuli": "Lapislazuli",
    "Obsidian": "Obsidian",
    "Crying Obsidian": "Weinender Obsidian",
    "Bedrock": "Grundgestein",
    "Deepslate": "Tiefenschiefer",
    "Cobblestone": "Bruchstein",
    "Glowstone": "Leuchtstein",
    # Mobs & Entities
    "Creeper": "Creeper",
    "Enderman": "Enderman",
    "Ender Dragon": "Enderdrache",
    "Wither": "Wither",
    "Warden": "Warden",
    "Phantom": "Phantom",
    "Ravager": "Verwüster",
    "Pillager": "Plünderer",
    "Evoker": "Magier",
    "Vindicator": "Diener",
    "Witch": "Hexe",
    "Blaze": "Lohe",
    "Ghast": "Ghast",
    "Shulker": "Shulker",
    "Skeleton": "Skelett",
    "Zombie": "Zombie",
    "Drowned": "Ertrunkener",
    "Husk": "Wüstenzombie",
    "Stray": "Eiswanderer",
    "Piglin": "Piglin",
    "Hoglin": "Hoglin",
    "Allay": "Hilfsgeist",
    "Axolotl": "Axolotl",
    "Villager": "Dorfbewohner",
    "Iron Golem": "Eisengolem",
    # Items & Mechanics
    "Ender Pearl": "Enderperle",
    "Eye of Ender": "Enderauge",
    "Elytra": "Elytren",
    "Totem of Undying": "Totem der Unsterblichkeit",
    "Enchanted Golden Apple": "Verzauberter goldener Apfel",
    "Golden Carrot": "Goldene Karotte",
    "Brewing Stand": "Braustand",
    "Anvil": "Amboss",
    "Enchanting Table": "Zaubertisch",
    "Smithing Table": "Schmiedetisch",
    "Beacon": "Leuchtfeuer",
    "Conduit": "Aktivierter Meeresleiter",
    "Shulker Box": "Shulkerkiste",
    "Ender Chest": "Endertruhe",
    "Respawn Anchor": "Seelenanker",
    "Lead": "Leine",
    "Name Tag": "Namensschild",
    "Saddle": "Sattel",
    "Crossbow": "Armbrust",
    "Trident": "Dreizack",
    "Shield": "Schild",
    "Spyglass": "Fernrohr",
    "Fishing Rod": "Angel",
    "Flint and Steel": "Feuerzeug",
    "Spawn Egg": "Spawn-Ei",
    # Dimensions & Structures
    "The End": "Das Ende",
    "Nether": "Der Nether",
    "Overworld": "Die Oberwelt",
    "Stronghold": "Festung",
    "Nether Fortress": "Netherfestung",
    "Bastion Remnant": "Bastionsruine",
    "End City": "Endstadt",
    "Woodland Mansion": "Waldanwesen",
    "Ocean Monument": "Ozeanmonument",
    "Ancient City": "Antike Stadt",
    # Mod / Server Mechanics
    "LoW": "Legend of War (Server Mod)",
    "Ansturm": "Rush / Charge Ability",
    "Doppelsprung": "Double Jump Ability",
    "dc": "Discord / Disconnected",
    "tp": "Teleport",
    "grief": "Griefing / Zerstörung von Bauten",
    "base": "Basis / Stützpunkt",
    "vanilla": "Vanilla (unmodifiziertes Spiel)"
}


def build_bilingual_glossary():
    print(f"--> Reading raw glossary: {INPUT_GLOSSARY}")
    all_terms = set()
    if os.path.exists(INPUT_GLOSSARY):
        with open(INPUT_GLOSSARY, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for length_key, terms in data.items():
                    for t in terms:
                        t_clean = t.strip()
                        if t_clean and not t_clean.isdigit() and len(t_clean) > 1:
                            all_terms.add(t_clean)
            except Exception as e:
                print(f"[WARN] Error reading JSON glossary: {e}")

    print(f"    Found {len(all_terms)} unique Minecraft concepts.")

    # Merge base translations
    bilingual_dict = dict(EN_TO_DE_BASE)

    # For any extra items, if they don't have an explicit DE translation, keep them as standard terms
    for term in all_terms:
        if term not in bilingual_dict:
            bilingual_dict[term] = term  # Many gamer terms stay unchanged

    os.makedirs(os.path.dirname(OUTPUT_GLOSSARY), exist_ok=True)
    with open(OUTPUT_GLOSSARY, "w", encoding="utf-8") as f:
        json.dump(bilingual_dict, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Built bilingual glossary with {len(bilingual_dict)} terms -> {OUTPUT_GLOSSARY}")


if __name__ == "__main__":
    build_bilingual_glossary()
