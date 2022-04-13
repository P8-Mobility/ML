class WordPhonemeMap:
    """
    Wrapper class for a dictionary mapping words to their phonemes
    """
    word_phoneme_map: dict[str, str] = {
        "paere": "pʰ æː ɐ",
        "baere": "p æː ɐ",
        "laere": "l æː ɐ",
        "skaere": "s k æː ɐ",
        "paererne": "pʰ æː ɐ n ə",
        "baekken": "p ɛ k ŋ",
        "udtale": "u ɤ t s æː l ə",
        "pony": "pʰ ɒ n i",
        "putin": "pʰ u t e n",
        "aeer": "æː ɐ",
        "er": "æ ɐ"
    }

    @staticmethod
    def get(word) -> str:
        """
        Returns the phonemes related to the word, seperated by spaces.
        If the word is not present in the map, an empty string is returned

        :param word: the word to get the phonemes of
        :return: the phonemes related to the word or an empty string if the word is not present in the map
        """
        if word not in WordPhonemeMap.word_phoneme_map:
            return ""
        return WordPhonemeMap.word_phoneme_map.get(word)

    @staticmethod
    def contains(word) -> bool:
        """
        Checks whether the given word is present in the map

        :param word: the word to check for in the map
        :return: whether the word is present in the map
        """
        return word in WordPhonemeMap.word_phoneme_map
