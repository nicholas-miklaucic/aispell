#!/usr/bin/env sh

# Processes the custom wordlist into SymSpell's format.
sed 's/$/ 1/' data/custom_wordlist.txt > data/custom_dictionary.txt
