#GUI for the project.

from sklearn.externals import joblib
from Tkinter import *
top = Tk()
L1 = Label(top, text="User Name")
L1.pack( side = LEFT)
E1 = Entry(top, bd =5)
E1.pack(side = RIGHT)

L2 = Label(top, text="User Name")
L2.pack( side = LEFT)
E2 = Entry(top, bd =5)
E2.pack(side = RIGHT)

L3 = Label(top, text="User Name")
L3.pack( side = LEFT)
E3 = Entry(top, bd =5)
E3.pack(side = RIGHT)

L4 = Label(top, text="User Name")
L4.pack( side = LEFT)
E4 = Entry(top, bd =5)
E4.pack(side = RIGHT)

L5 = Label(top, text="User Name")
L5.pack( side = LEFT)
E5 = Entry(top, bd =5)
E5.pack(side = RIGHT)

L6 = Label(top, text="User Name")
L6.pack( side = LEFT)
E6 = Entry(top, bd =5)
E6.pack(side = RIGHT)

L7 = Label(top, text="User Name")
L7.pack( side = LEFT)
E7 = Entry(top, bd =5)
E7.pack(side = RIGHT)

L8 = Label(top, text="User Name")
L8.pack( side = LEFT)
E8 = Entry(top, bd =5)
E8.pack(side = RIGHT)

L9 = Label(top, text="User Name")
L9.pack( side = LEFT)
E9 = Entry(top, bd =5)
E9.pack(side = RIGHT)

L10 = Label(top, text="User Name")
L10.pack( side = LEFT)
E10 = Entry(top, bd =5)
E10.pack(side = RIGHT)

L11 = Label(top, text="User Name")
L11.pack( side = LEFT)
E11 = Entry(top, bd =5)
E11.pack(side = RIGHT)

L12 = Label(top, text="User Name")
L12.pack( side = LEFT)
E12 = Entry(top, bd =5)
E12.pack(side = RIGHT)

L13 = Label(top, text="User Name")
L13.pack( side = LEFT)
E13 = Entry(top, bd =5)
E13.pack(side = RIGHT)

L14 = Label(top, text="User Name")
L14.pack( side = LEFT)
E14 = Entry(top, bd =5)
E14.pack(side = RIGHT)

L15 = Label(top, text="User Name")
L15.pack( side = LEFT)
E15 = Entry(top, bd =5)
E15.pack(side = RIGHT)

clf = joblib.load('clf_o.pkl')

top.mainloop()