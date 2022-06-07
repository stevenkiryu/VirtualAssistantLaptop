import sqlite3 as sql
from flask import Flask, render_template , request

App = Flask(__name__)

@App.route("/")
def route():
    connect_laptop = sql.connect('Dataset Laptop.db')
    print("Yay. Connection Complete")

    select_home_laptop = connect_laptop.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop FROM DatasetLaptop")

    row = select_home_laptop.fetchall()

    return render_template("Home_test.html", rows=row)

if __name__ == "__main__":
    App.run()


