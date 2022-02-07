from transformers import AutoTokenizer
from translator.preprocess import group_to_fit

example_cs = [
    "Ministerstvo financí se tedy dostalo do nezáviděníhodné pozice, když muselo komunistické vrchnosti vysvětlovat, že jejich bůh je falešný, že jejich dokonalý teoretický model se prostě do reálného světa nehodí a že by se měli přizpůsobit realitě, již odvrhli",
    "Úředníci na ministerstvu byli většinou pozorovatelé, připomínali děti, které hrají nějakou počítačovou hru, jíž sice nevěří, ale přesto je baví",
    "Někteří z nich byli velmi inteligentní, takže se do hry sami zapojovali a občas na svých obchodech a transakcích i vydělali",
    "Tito lidé pak získávali na ministerstvu stále vyšší a vyšší místa",
    "Někteří jezdili do práce svými vlastními automobily a přátelili se s nově vzniklou třídou místních průmyslníků, kteří spálili své ideologické svěrací kazajky a začali v komunistické společnosti pracovat jako kapitalisté",
    "Tím svému státu přinášeli bohatství, a získali si tak vlažnou vděčnost, ne-li úctu, ze strany politických vládců, kteří se k nim chovali asi jako k hodnému ovčáckému psovi",
]

example_en = [
    "The Ministry of Finance, therefore, was placed in the unenviable position of having to explain to the communist clergy that their god was a false one, that their perfect theoretical model just didn't play in the real world, and that therefore they had to bend to a reality which they had rejected",
    "The bureaucrats in the ministry were for the most part observers, rather like children playing a computer game that they didn't believe in but enjoyed anyway",
    "Some of the bureaucrats were actually quite clever, and played the game well, sometimes even making a profit on their trades and transactions",
    "Those who did so won promotions and status within the ministry",
    "Some even drove their own automobiles to work and were befriended by the new class of local industrialists who had shed their ideological straitjackets and operated as capitalists within a communist society",
    "That brought wealth into their nation, and earned the tepid gratitude, if not the respect, of their political masters, rather as a good sheepdog might",
]

expected = """
<CS> Ministerstvo financí se tedy dostalo do nezáviděníhodné pozice, když muselo komunistické vrchnosti vysvětlovat, že jejich bůh je falešný, že jejich dokonalý teoretický model se prostě do reálného světa nehodí a že by se měli přizpůsobit realitě, již odvrhli. Úředníci na ministerstvu byli většinou pozorovatelé, připomínali děti, které hrají nějakou počítačovou hru, jíž sice nevěří, ale přesto je baví
<EN> The Ministry of Finance, therefore, was placed in the unenviable position of having to explain to the communist clergy that their god was a false one, that their perfect theoretical model just didn't play in the real world, and that therefore they had to bend to a reality which they had rejected. The bureaucrats in the ministry were for the most part observers, rather like children playing a computer game that they didn't believe in but enjoyed anyway
---
<CS> Někteří z nich byli velmi inteligentní, takže se do hry sami zapojovali a občas na svých obchodech a transakcích i vydělali. Tito lidé pak získávali na ministerstvu stále vyšší a vyšší místa. Někteří jezdili do práce svými vlastními automobily a přátelili se s nově vzniklou třídou místních průmyslníků, kteří spálili své ideologické svěrací kazajky a začali v komunistické společnosti pracovat jako kapitalisté
<EN> Some of the bureaucrats were actually quite clever, and played the game well, sometimes even making a profit on their trades and transactions. Those who did so won promotions and status within the ministry. Some even drove their own automobiles to work and were befriended by the new class of local industrialists who had shed their ideological straitjackets and operated as capitalists within a communist society
---
<CS> Tím svému státu přinášeli bohatství, a získali si tak vlažnou vděčnost, ne-li úctu, ze strany politických vládců, kteří se k nim chovali asi jako k hodnému ovčáckému psovi
<EN> That brought wealth into their nation, and earned the tepid gratitude, if not the respect, of their political masters, rather as a good sheepdog might
""".strip()


def test_group_to_fit():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")

    groups = group_to_fit(example_cs, example_en, tokenizer, max_tokens=100)
    actual = "\n---\n".join(f"<CS> {cs_part}\n<EN> {en_part}" for cs_part, en_part in groups)

    assert expected == actual
