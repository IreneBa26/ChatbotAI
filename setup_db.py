import sqlite3
import json




classi = """
	CREATE TABLE classi (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		classe VARCHAR(50) NOT NULL
	);
"""

domande = """
CREATE TABLE domande(
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	domanda VARCHAR(255) NOT NULL,
	id_classe INTEGER NOT NULL,

	FOREIGN KEY (id_classe) REFERENCES classi (id)
);
"""

risposte = """
	CREATE TABLE risposte (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		risposta VARCHAR(255) NOT NULL,
		id_classe INTEGER NOT NULL,

		FOREIGN KEY (id_classe) REFERENCES classi (id)
	);
"""

database = "bot.db"
conn = sqlite3.connect(database)
cursor = conn.cursor()

for tabella in (classi, domande, risposte):
	cursor.execute(tabella)

with open("data.json") as f:
	data = json.load(f)
	for elemento in data:
		q = "INSERT INTO classi (classe) VALUES ('{}')"
		cursor.execute(q.format(elemento['classe']))
		id_classe = cursor.lastrowid

		for domanda in elemento['domande']:
			q= """
				INSERT INTO domande (domanda, id_classe)
				VALUES
					("{0}", "{1}")
			""".format(domanda, id_classe)
			cursor.execute(q)

		for risposta in elemento['risposte']:
			q = """
				INSERT INTO risposte (risposta, id_classe)
				VALUES 
					("{0}", "{1}")
			""".format(risposta, id_classe)
			cursor.execute(q)

conn.commit()