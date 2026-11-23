# Себестоимость 1 обработки SONYA по режиму и длине видео (консервативно). Данные: jobdur.tsv (74 задачи 12–19.11),
# fleet.db (dph 0.07–0.17 $/ч), ledger.db (факт Vast 13–19.11 ≈ $22 на ~67 задач с тестами), llm_usage (Gemini токены).
FX=82.19                      # ₽/$ ЦБ 21.11
GPU_USD_H=0.15                # верх диапазона живых серверов (3060/A4000 0.07–0.17)
BOOT_MIN=4                    # пробуждение/хвост простоя на задачу
NET_USD_GB=0.02               # трафик Vast ~$20/TB (кап 25)
CREATE_USD=0.05               # доля создания сервера (~$0.16 на create, ~1 create на 3 задачи в пике)
S3_RUB_GB_MONTH=2.0           # хранение 1 месяц
GEM={"viral":0.7,"hooks":1.0,"story":2.5,"edu":0.7,"streamer":1.0,"trailer":2.5,"beep":0.0,"gen":0.0}
# минуты GPU (p80 из jobdur.tsv после ускорений 15–17.11; длинные режутся по 1080p/Speedfix)
GPU_MIN={  # 30 / 60 / 180 мин исходника
 "viral":(10,15,25),"hooks":(12,18,30),"story":(10,15,28),"edu":(18,28,45),
 "streamer":(10,15,30),"trailer":(12,15,25),"beep":(6,11,32),"gen":(2,2,2)}
IN_GB={30:0.9,60:1.8,180:5.0}   # 1080p исходник
OUT_GB={"viral":0.35,"hooks":0.3,"story":0.6,"edu":0.2,"streamer":0.3,"trailer":0.2,"beep":None,"gen":0.01}
NAMES={"viral":"Вирал","hooks":"Хуки","story":"Сторителлинг","edu":"Образовательный","streamer":"Стример","trailer":"Трейлер","beep":"BEEP","gen":"Gen (картинки)"}
def cost(m,dur):
    i=(30,60,180).index(dur); gm=GPU_MIN[m][i]+(0 if m=="gen" else BOOT_MIN)
    gpu=gm/60*GPU_USD_H*FX
    if m=="gen": return gpu, dict(gpu=gpu)
    inn=IN_GB[dur]; out=OUT_GB[m] if OUT_GB[m] is not None else inn*0.9
    net=(inn+out)*NET_USD_GB*FX; create=CREATE_USD*FX; s3=(inn+out)*S3_RUB_GB_MONTH
    tot=gpu+net+create+s3+GEM[m]
    return tot, dict(gpu=gpu,net=net,create=create,s3=s3,gem=GEM[m])
if __name__=="__main__":
    print("%-16s %8s %8s %8s"%("режим","30мин","60мин","180мин"))
    tot={30:0,60:0,180:0}
    for m in NAMES:
        c=[cost(m,d)[0] for d in (30,60,180)]
        for d,x in zip((30,60,180),c): tot[d]+=x*10
        print("%-16s %8.1f %8.1f %8.1f"%(NAMES[m],*c))
    print("10 в каждом из 8:", {d:round(v) for d,v in tot.items()})
    pub=["viral","hooks","story","edu","streamer","beep"]
    t6={d:round(sum(cost(m,d)[0]*10 for m in pub)) for d in (30,60,180)}
    print("10 × 6 открытых режимов (без Трейлера и Gen):",t6)
    print("разбивка Вирал 60:",{k:round(v,1) for k,v in cost("viral",60)[1].items()})
    print("разбивка Стример 180:",{k:round(v,1) for k,v in cost("streamer",180)[1].items()})
