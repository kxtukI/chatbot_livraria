import json
from tkinter import *
from extract import class_prediction, get_response

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model

model = load_model('model.keras')

with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

base = Tk()
base.title("Chatbot - Livraria")
base.geometry("400x500") 
base.resizable(width=FALSE, height=FALSE)

def chatbot_response(msg):
    """
    Resposta do bot
    """
    ints = class_prediction(msg, model)
    res = get_response(ints, intents)
    return res

def send(event=None):
    """
    Envia a mensagem
    """
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        Chat.config(state=NORMAL)
        Chat.insert(END, f"Você: {msg}\n\n")
        Chat.config(foreground="#000000", font=("Arial", 12))

        response = chatbot_response(msg)
        Chat.insert(END, f"Bot: {response}\n\n")

        Chat.config(state=DISABLED)
        Chat.yview(END)

def clear_chat():
    """
    Limpa toda a conversa
    """
    Chat.config(state=NORMAL)
    Chat.delete('1.0', END)
    Chat.config(state=DISABLED)

Chat = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", wrap=WORD)
Chat.config(state=DISABLED, insertofftime=0, insertontime=0)

scrollbar = Scrollbar(base, command=Chat.yview)
Chat['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 10, 'bold'), text="Enviar", width="12", height=2, bd=0, 
                    bg="#666", activebackground="#333", fg='#ffffff', command=send)

ClearButton = Button(base, font=("Verdana", 10, 'bold'), text="Limpar", width="12", height=2, bd=0, 
                     bg="#999", activebackground="#666", fg='#ffffff', command=clear_chat)

EntryBox = Text(base, bd=0, bg="white", width="29", height="2", font="Arial")

EntryBox.bind('<Return>', send)

scrollbar.place(x=376, y=6, height=386)
Chat.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=50, width=260)
SendButton.place(x=270, y=401, height=50, width=100)
ClearButton.place(x=270, y=455, height=40, width=100)

base.mainloop()