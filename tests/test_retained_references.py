import pytest

from jieyi.quality.references import is_retained_reference


@pytest.mark.parametrize("text", [
    "(Paul Éluard, Poésie involontaire et poésie intentionnelle.)",
    "Jones, David. 2004. Reforming the morality of usury: a study of the social space of money.",
    "41. Hornborg 2005; Hornborg and Hill (eds) 2011.",
    "—. 2008. ‘Des crânes et des vautours ou la guerre oubliée.’ Paléorient 34 (1): 33–58.",
    "49. Nelida Fuccaro, Histories of City and State in the Persian Gulf.",
    "Davidson, Christopher M. “Arab Nationalism and British Opposition in Dubai, 1920–66.” Middle Eastern Studies 43, no. 6 (November 2007): 879–90.",
    "Anscombe, Frederick, 228n13, 228n15",
    "144. Aubin, “Le royaume d’Ormuz,” 150–51.",
    "192. D. T. Potts, “From Qadê to Mazûn: Four notes on Oman, c. 700 B.C. to 700 A.D.,” JOS 8 (1985): 81–95.",
    'Chittolini, Giorgio, 1989 "Cities, City-States and Regional States in North Italy."',
    "Darwin 1859.",
    "Darwin and Wallace 1858; Darwin 1859.",
    "Smith, John, 14–18, 23, 29n3.",
    "(Paul Éluard,\n Poésie involontaire et poésie intentionnelle.)",
    "27. Rugh, The Political Culture of Leadership in the United Arab Emirates, 103, 113.",
    "Goswami, Manu. Producing India: From Colonial Economy to National Space. Chicago: University of Chicago Press, 2004.",
    "cartography, 58–61, 197, 218n50. See also maps",
    "2. D. T. Potts, The Arabian Gulf in Antiquity, vol. 2 (Oxford: Clarendon Press, 1990), 135.",
    "and Early Islamic Near East, II: Land Use and Settlement Patterns, ed. G. King and A. Cameron (Princeton, NJ: Darwin Press, 1994), 38.",
    "125. “Revenue of the kingdom of Hormuz,” undated [1515], in António Dias Farinha, “Os portugueses no golfo Pérsico,” 38.",
    "and Barry Maguire (eds.), Weighing Reasons (Oxford: Oxford University Press), 213–40.",
    "-- 1974 Massenannut und Hungerkrisen im vorindustriellen Europa. Hamburg and Berlin: Paul Parey.",
    "(Brouillons de Bouvard et Pécuchet, plan final, Pléiade, II, 986)",
])
def test_retained_bibliographic_or_index_entries(text):
    assert is_retained_reference(text)


@pytest.mark.parametrize("text", [
    "How bow'd the woods beneath their sturdy stroke!",
    "In 2004, David Jones wrote about the morality of usury.",
    "David Jones argued in 2004 that money changes social relationships.",
    "Hornborg and Hill discussed these problems in 2011.",
    "41. Hornborg argued that power was distributed unequally in 2005.",
    "2004 was a difficult year for the city and its people.",
    "(Paul Éluard, a poet who lived in France.)",
    "49. Nelida Fuccaro, a historian, studied the Persian Gulf.",
    "49. Nelida Fuccaro, Histories of City and State. This changes the argument.",
    "Darwin wrote, ‘There is grandeur in this view of life.’",
    "192. They described four notes on Oman in 1985: the account was incomplete.",
    "The index includes Anscombe, Frederick, and other philosophers.",
    "John Smith, 2004, was living beside the river.",
    "The woods were silent, 1751.",
    "Thomas Gray",
    "",
])
def test_prose_and_dates_are_not_exempted(text):
    assert not is_retained_reference(text)
