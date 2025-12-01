# Receptų Generatorius

Tai programa, kuri pagal tavo turimus produktus suranda geriausius receptus naudodama **Neuroninį tinklą**. Ji taip pat leidžia filtruoti pagal dietas (Vegan, Keto ir t.t.), vengti alergenų ir sugeneruoti patiekalo nuotrauką naudojant AI.

## Ko reikia
*   Python
*   Suinstaliuotos bibliotekos:
    ```bash
    pip install -r requirements.txt
    ```

## Paruošimas (Vykdyti tik pirmą kartą)
Prieš naudojant programą, reikia išpakuoti duomenis ir apmokyti modelį:

1.  **Išpakuokite failą `full_dataset.zip`** į šį aplanką (turi atsirasti failas `full_dataset.csv`).
2.  Duomenų paruošimas:
    ```bash
    python prepare_ml_data.py
    ```
3.  Modelio apmokymas:
    ```bash
    python ModelTrain.py
    ```

## Paleidimas
Norėdami paleisti programą, tiesiog atidarykite failą:
**`run.bat`**
