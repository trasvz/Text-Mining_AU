from flask import Flask,render_template, request,jsonify
import pickle

import re

import json 
import nltk
import string  
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

with open("tv.pkl", "rb") as fp:
        tv = pickle.load(fp)

with open("logistic regression.pkl", "rb") as fp:
        lgr = pickle.load(fp)

import requests

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        # input_lang = request.form['input_lang']
        bahasaoutput = request.form['bahasaoutput']
        sms = request.form['message']

        hasil = predict(sms)
        hasil = str(hasil)
        hasil = hasil[2:-2]

        data={
                # "judul": '\"'+judul+'\"',
                "isi": '\"'+sms+'\"',
                # "instansi" : '\"'+nama_instansi+'\"',
                "kelasasing": '\"'+hasil+'\"',
                "SourceLanguageCode": '\"id\"',
                "TargetLanguageCode": '\"'+bahasaoutput+'\"'
            }
        data_translated=translation(data)
        data_translated['TargetLanguageCode'] = encode(data_translated['TargetLanguageCode'])        
        return render_template("output.html",pesanawal=sms, prediksi=hasil, data_translated=data_translated)
    else:
        return render_template("index.html")

def normalize_document(kalimat):
    kalimat = str(kalimat).lower()
    kalimat = re.sub(r'[^a-z\s]', '', kalimat)
    kalimat = kalimat.strip()
    
    stop = stopword.remove(kalimat)
    hasil = stemmer.stem(stop)
    tokens = nltk.tokenize.word_tokenize(hasil)
    
    doc = ' '.join(tokens)
    return doc
        
def encode(lang):
    encoding={
        "af":"Afrikaans",
        "sq":"Albanian",
        "am":"Amharic - አማርኛ",
        "ar":"Arabic - العربية",
        "hy":"Armenian - հայերեն",
        "as":"Assamese",
        "az":"Azerbaijani - azərbaycan dili",
        "ba":"Bashkir",
        "eu":"Basque - euskara",
        "be":"Belarusian - беларуская",
        "bn":"Bengali - বাংলা",
        "bs":"Bosnian - bosanski",
        "bg":"Bulgarian - български",
        "my":"Burmese",
        "ca":"Catalan - català",
        "ceb":"Cebuano",
        "km":"Central Khmer",
        "zh":"Chinese - 中文",
        "zh-TW":"Chinese (Traditional) - 中文（繁體）",
        "cv":"Chuvash",
        "hr":"Croatian - hrvatski",
        "cs":"Czech - čeština",
        "da":"Danish - dansk",
        "nl":"Dutch - Nederlands",
        "en":"English",
        "eo":"Esperanto - esperanto",
        "et":"Estonian - eesti",
        "fi":"Finnish - suomi",
        "fr":"French - français",
        "gl":"Galician - galego",
        "ka":"Georgian - ქართული",
        "de":"German - Deutsch",
        "el":"Greek - Ελληνικά",
        "gu":"Gujarati - ગુજરાતી",
        "ht":"Haitian",
        "he":"Hebrew - עברית",
        "hi":"Hindi - हिन्दी",
        "hu":"Hungarian - magyar",
        "is":"Icelandic - íslenska",
        "ilo":"Iloko",
        "id":"Indonesian - Indonesia",
        "ga":"Irish - Gaeilge",
        "it":"Italian - italiano",
        "ja":"Japanese - 日本語",
        "jv":"Javanese",
        "kn":"Kannada - ಕನ್ನಡ",
        "kk":"Kazakh - қазақ тілі",
        "ky":"Kirghiz",
        "ko":"Korean - 한국어",
        "ku":"Kurdish - Kurdî",
        "la":"Latin",
        "lv":"Latvian - latviešu",
        "lt":"Lithuanian - lietuvių",
        "lb":"Luxembourgish",
        "mk":"Macedonian - македонски",
        "ms":"Malay - Bahasa Melayu",
        "ml":"Malayalam - മലയാളം",
        "mr":"Marathi - मराठी",
        "mn":"Mongolian - монгол",
        "ne":"Nepali - नेपाली",
        "no":"Norwegian - norsk",
        "or":"Oriya - ଓଡ଼ିଆ",
        "ps":"Pashto - پښتو",
        "fa":"Persian - فارسی",
        "pl":"Polish - polski",
        "pt":"Portuguese - português",
        "pa":"Punjabi - ਪੰਜਾਬੀ",
        "qu":"Quechua",
        "ro":"Romanian - română",
        "ru":"Russian - русский",
        "sa":"Sanskrit",
        "gd":"Scottish Gaelic",
        "sr":"Serbian - српски",
        "sd":"Sindhi",
        "si":"Sinhala - සිංහල",
        "sk":"Slovak - slovenčina",
        "sl":"Slovenian - slovenščina",
        "so":"Somali - Soomaali",
        "es":"Spanish - español",
        "su":"Sundanese",
        "sw":"Swahili - Kiswahili",
        "sv":"Swedish - svenska",
        "tl":"Tagalog",
        "tg":"Tajik - тоҷикӣ",
        "ta":"Tamil - தமிழ்",
        "tt":"Tatar",
        "te":"Telugu - తెలుగు",
        "th":"Thai - ไทย",
        "tr":"Turkish - Türkçe",
        "tk":"Turkmen",
        "uk":"Ukrainian - українська",
        "ur":"Urdu - اردو",
        "ug":"Uyghur",
        "uz":"Uzbek - o‘zbek",
        "vi":"Vietnamese - Tiếng Việt",
        "cy":"Welsh - Cymraeg",
        "yi":"Yiddish",
        "yo":"Yoruba - Èdè Yorùbá"
        } 
    return encoding[lang]

def predict(sms):
    doc = normalize_document(sms)
    vektor = tv.transform([doc])
    predicted_label = lgr.predict(vektor)
    return(predicted_label)

def translation(data):

    isi="{\r\n"+"  \"isi\": "+data['isi']+",\r\n"
    kelasasing="  \"kelasasing\": "+data['kelasasing']+",\r\n"
    # instansi="  \"instansi\":"+data['instansi']+",\r\n"
    sourcelang=" \"SourceLanguageCode\":"+data['SourceLanguageCode']+",\r\n"
    targetlang=" \"TargetLanguageCode\":"+data['TargetLanguageCode']+"\r\n}"
    url = "https://1u5h3d6jef.execute-api.us-east-1.amazonaws.com/Comprehend_Translate"
    headers = {
    'Content-Type': 'text/plain',
    'charset':'utf-8',
    }

    response = requests.request("GET", url, headers=headers, data=(isi+kelasasing+sourcelang+targetlang).encode('utf-8'))
    return(response.json())

# @app.route('/api/', methods=['GET'])
# def api():
#     query_parameters = request.args
#     # judul_raw=query_parameters['judul']
#     kelasasing_raw=query_parameters['kelasasing']
#     isi_raw=query_parameters['isi']
#     input_lang=query_parameters['input_lang']
#     output_lang=query_parameters['bahasaoutput']
#     if(input_lang!='id'):
#         # data_input={
#         #     "judul": '\"'+judul_raw+'\"',
#         #     "isi": '\"'+isi_raw+'\"',
#         #     "instansi" : '\"'+'blank'+'\"',
#         #     "SourceLanguageCode": '\"'+input_lang+'\"',
#         #     "TargetLanguageCode": '\"id\"'
#         # }
#         # data_translated=translation(data_input)
#         # judul = data_translated['judul_translated']
#         # isi = data_translated['isi_translated']
#     else:
#         # judul = judul_raw
#         isi = isi_raw
#         kelasasing = kelasasing_raw
#     nama_instansi = predict(judul,isi)
#     data={
#             "judul": '\"'+judul+'\"',
#             "isi": '\"'+isi+'\"',
#             "instansi" : '\"'+nama_instansi+'\"',
#             "SourceLanguageCode": '\"id\"',
#             "TargetLanguageCode": '\"'+output_lang+'\"'
#         }
#     data_translated=translation(data)
#     print(data_translated)
#     hasil_akhir={
#         'data_input':{
#             'Title':judul_raw,
#             'Content':isi_raw,
#             'Input Language':input_lang
#         },
#         'data_translated':{
#             'Title':data_translated['judul_translated'],
#             'Content':data_translated['isi_translated'],
#             'Government Agency':data_translated['instansi_translated'],
#             'Output Language':data_translated['TargetLanguageCode']
#         }
#     }
#     return jsonify(hasil_akhir)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080')
    # app.run(debug="True")