# MLA – mindre minne för attention

MLA (Multi-head Latent Attention) minskar mängden data som en språkmodell behöver spara när den genererar text. I stället för att spara separata keys och values för varje head sparar den ett gemensamt kort per token och lager, tillsammans med en liten positionsdel. Med absorption kan modellen använda korten utan att först återskapa varje gammal tokens fullständiga keys och values. Här går jag igenom hur det fungerar och använder en liten demo för att jämföra minnesbehov och räknearbete.

## 1. Vad modellen gör

För att en språkmodell ska kunna producera text utgår den från texten som redan har presenterats. Texten delas upp i tokens, som kan vara hela ord, delar av ord eller skiljetecken. Utifrån den tidigare texten räknar modellen fram sannolikheter för vilken token som kan komma härnäst. Den väljer sedan en token, lägger den sist i texten och gör samma procedur igen.

Varje token får ett ID som används för att hämta en lista med tal, alltså en vektor. Den första representationen kallas en embedding. I modellen finns flera lager efter varandra som bearbetar vektorerna och låter information från sammanhanget påverka dem. Representationen av en token vid ett visst lager kallas hidden state och skrivs som h. Samma token kan därför få olika representationer beroende på vilken text den finns i.

En stor del av arbetet i ett lager är matrismultiplikation, W * h. W är en tabell med tal som modellen lärde sig under träningen. Varje rad i tabellen bestämmer hur talen i h ska multipliceras och läggas ihop för att skapa ett tal i den nya listan. Resultatet blir en ny vektor.

## 2. Attention och KV-cachen

Tänk att modellen har läst "Lisa tog på sig jackan. Hon".

För varje position har modellen redan skapat två saker: en key, som fungerar som en söketikett, och en value, som innehåller information positionen kan bidra med.

När modellen kommer till "hon" skapar den en query. Vi kan föreställa oss den som frågan: "Vem i den tidigare texten kan det här syfta på?"

Queryn jämförs med de sparade etiketterna. Genom träningen har modellen lärt sig beräkningar som kan ge en hög matchning mellan "hon" och "Lisa". Då får informationen från Lisa större vikt i blandningen som modellen använder vidare.

Det finns inga färdigskrivna frågor eller etiketter inne i modellen. Allt är listor med tal. Liknelsen beskriver vad beräkningarna används till.

Modellen sparar keys och values eftersom de kommer att behövas igen när texten fortsätter. Det är dessa sparade listor som kallas KV-cachen.

## 3. MLA: ett kort per token och absorption

Vanlig attention sparar mycket eftersom varje head har sin egen key och value för varje token. Med 128 heads och 128 tal i varje key och value blir det 32 768 tal för en enda token i ett lager. Samma sak behövs i alla attention-lager och för alla tokens i texten. Därför växer minnesbehovet snabbt.

MLA sparar i stället ett gemensamt kort med 512 tal per token och lager, plus en liten positionsdel. Alla heads använder samma kort, men har egna inlärda sätt att läsa informationen från det. Modellen tränas att skapa och använda korten från början.

Man skulle kunna packa upp varje kort till varje heads key och value när de behövs. Men då måste modellen göra om den uppackningen för alla gamla tokens varje gång den fortsätter skriva. Det skulle innebära mycket extra arbete.

Absorption löser detta genom att ändra hur beräkningen utförs. I stället för att först omvandla varje kort till en key, omvandlar headet sin query så att den kan jämföras direkt med kortet. Matematiken ger samma matchningspoäng i båda fallen.

Den omvandlade queryn kan sedan användas mot alla sparade kort. Headet behöver alltså göra en omvandling av sin nya fråga, i stället för att återskapa alla gamla keys.

På value-sidan kan modellen först blanda korten med attention-vikterna och sedan omvandla den sammanlagda informationen. Den behöver därför inte återskapa varje gammal tokens value separat.

## 4. Position

Modellen behöver också veta var orden står i texten, och RoPE (rotary position embedding) ger positionsinformation genom att rotera query och key beroende på position.

Att rotera kortet och sedan packa upp det ger i allmänhet inte samma resultat som att först packa upp det och sedan rotera keyn, eftersom ordningen mellan omvandlingarna spelar roll. MLA låter därför innehållskortet vara oroterat och använder en separat key med 64 tal för positionsdelen, delad av alla heads. Då kan innehållet fortfarande jämföras direkt mot korten med absorption, och poängen från positionsjämförelsen läggs till efteråt.

## 5. Vad det ger och vad det kostar

Siffrorna gäller DeepSeek-V2: 128 heads med 128 tal i varje key och value, kort med 512 tal, en positions-key med 64 tal och 60 lager. Minnet är räknat med 2 byte per tal.

| | Vanlig attention | MLA |
|---|---:|---:|
| Sparade tal per token och lager | 32 768 | 576 |
| Minne per token, alla lager | 3,9 MB | 69,1 kB |
| Cache för 131 072 tokens | 515,4 GB | 9,1 GB |

Räknearbete för attention i ett lager per ny token, mätt i multiply-adds. En multiply-add är en multiplikation följd av en addition.

| Tokens i cachen | Vanlig attention | MLA med uppackning | MLA med absorption |
|---:|---:|---:|---:|
| 32 | 1,05 miljoner | 538 miljoner | 21,0 miljoner |
| 4 096 | 134 miljoner | 68,9 miljarder | 554 miljoner |
| 131 072 | 4,29 miljarder | 2 200 miljarder | 17,2 miljarder |

Tabellerna visar att MLA i den här jämförelsen minskar cachen från 32 768 till 576 tal per token och lager, alltså ungefär 57 gånger mindre lagrad data. Priset är mer räknearbete – i tabellens längsta exempel gör MLA med absorption ungefär fyra gånger så många multiply-adds som vanlig attention. Absorption behövs ändå för att slippa packa upp alla gamla kort vid varje ny token, vilket annars skulle ge mycket mer räknearbete.

## 6. Kör demon

Kör från projektets rotmapp:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python mla_demo.py
.venv/bin/python -m pytest
```

Demon börjar med ett leksaksexempel med ett head, fyra kort och heltalsvikter. Båda vägarna ger samma poäng och samma resultat:

```text
cards                    [[3, 1], [1, 1], [2, 2], [1, 2]]
query                    [1, 2]
scores, keys rebuilt     [6, 4, 8, 7]
scores, query translated [6, 4, 8, 7]
output, unpack           [1.835, 3.734]
output, absorbed         [1.835, 3.734]
same answer: yes
```

Hela utskriften finns i [results/demo_output.txt](results/demo_output.txt).

## 7. Källor

- Vaswani m.fl., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017. Avsnitt 3.2 beskriver attention, query, key, value och heads.
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024. Avsnitt 2.1.2 beskriver kortet och absorption, avsnitt 2.1.3 positionsdelen och avsnitt 3.1.2 modellens storlekar.
