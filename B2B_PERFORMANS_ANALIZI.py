import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

warnings.filterwarnings("ignore")
st.set_page_config(page_title="B2B Satış Dashboard", layout="wide")


@st.cache_data
def veri_oku():
    df = pd.read_excel("B2B_Transaction_Data_.xlsx")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    for col in ["InvoiceNo", "CustomerID", "Description", "Category", "City"]:
        df[col] = df[col].astype(str).where(df[col].notna())
    return df


EKSIK_KOLONLAR = ["Category", "City", "Quantity"]
ELE = "Eksik Değere Sahip Olan İşlemleri Ele"

def imputation(ham, secim):
    df = ham.copy()
    doldurulan = None

    if secim == ELE:
        df = df.dropna(subset=EKSIK_KOLONLAR)
    else:
        imputer = SimpleImputer(strategy="most_frequent")
        df[EKSIK_KOLONLAR] = imputer.fit_transform(df[EKSIK_KOLONLAR].astype(object))
        df["Quantity"] = df["Quantity"].astype(float)
        doldurulan = imputer.statistics_.astype(str)     
         
    df["Satis"] = df["Quantity"] * df["NetPrice"]
    df["Indirim"] = df["Quantity"] * (df["UnitPrice"] - df["NetPrice"])
    df["Ay"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Yıl"] = df["InvoiceDate"].dt.year.astype(str)
    return df, doldurulan


DOLDUR = "Eksik Değere Sahip Olanları En Çok Geçen Değer ile Doldur"

def imputation_paneli(ham):
    with st.expander("Eksik Değer Yönetimi", expanded=True):
        secim = st.radio("Eksik değerler nasıl ele alınsın?", [ELE, DOLDUR], key="eksik_secim")

        df, doldurulan = imputation(ham, secim)
        etkilenen = int(ham[EKSIK_KOLONLAR].isna().any(axis=1).sum())  
        sol, sag = st.columns(2)
        sol.write("**Kolon bazında eksik değer adedi**")
        eksik = ham[EKSIK_KOLONLAR].isna().sum()

        if secim == ELE:
            sol.dataframe(pd.DataFrame({"İşlem Öncesi Eksik": eksik, "Elenen Adet": eksik}))
        else:
            sol.dataframe(pd.DataFrame({"İşlem Öncesi Eksik": eksik, "Doldurulan Adet": eksik, "Doldurulan Değer": doldurulan}))
            
        sag.write("**Veri setinin güncel durumu**")
        k1, k2, k3 = sag.columns(3)
        k1.metric("İlk Satır Sayısı", f"{len(ham):,}")
        k2.metric("Etkilenen Satır", f"{etkilenen:,}")
        k3.metric("Güncel Satır Sayısı", f"{len(df):,}")
        islem = "silindi" if secim == ELE else "en çok geçen değer ile dolduruldu"
        sag.info(f"Eksik değer içeren {etkilenen:,} satır {islem}.")
    return df


def oran_ekle(m):
    m["Müşteri Başına Satış"] = m["Satış Tutarı"] / m["Müşteri Adedi"]
    m["İşlem Başına Satış"] = m["Satış Tutarı"] / m["İşlem Adedi"]
    m["İşlem Başına Ürün Adedi"] = m["Ürün Adedi"] / m["İşlem Adedi"]
    return m


def metrikler(df, grup):
    g = df.groupby(grup)
    return oran_ekle(pd.DataFrame({
        "Satış Tutarı": g["Satis"].sum(),
        "İşlem Adedi": g["InvoiceNo"].nunique(),
        "Ürün Adedi": g["Quantity"].sum(),
        "İndirim Tutarı": g["Indirim"].sum(),
        "Müşteri Adedi": g["CustomerID"].nunique(),
        "Farklı Ürün Sayısı": g["Description"].nunique(),
        "Farklı Kategori Sayısı": g["Category"].nunique(),
        "Farklı İl Sayısı": g["City"].nunique(),
    }))


def toplam_metrikler(df):
    """Tüm veri için tek satırlık metrik özeti."""
    return metrikler(df, np.zeros(len(df))).iloc[0]


def kartlar(ozet, baslik_eki=""):
    """Metrikleri 4'lü sıralar halinde kart olarak gösterir."""
    for i in range(0, len(ozet), 4):
        for kolon, (ad, deger) in zip(st.columns(4), ozet.iloc[i:i + 4].items()):
            kolon.metric(ad + baslik_eki, f"{deger:,.2f}")


def tum_grafikler(tablo_sozlugu):
    """Her metrik için küçük bir trend grafiği çizer (3 kolon halinde)."""
    kolonlar = st.columns(3)

    for i, (ad, veri) in enumerate(tablo_sozlugu.items()):
        kolonlar[i % 3].caption(f"**{ad}**")
        kolonlar[i % 3].line_chart(veri, height=200)


def n_sec(yer, etiket, ust, varsayilan, key=None):
    """Açılır liste, varsayılan değerle başlar."""
    return yer.selectbox(etiket, range(1, ust + 1), index=min(varsayilan, ust) - 1, key=key)


def pay_ekle(tablo, deger, grup=None):
    """Değerin toplam (ya da grup toplamı) içindeki payını '%' metni olarak ekler."""
    toplam = tablo.groupby(grup)[deger].transform("sum") if grup else tablo[deger].sum()
    return tablo.assign(Pay=(100 * tablo[deger] / toplam).round(1).astype(str) + "%")


def pasta(tablo, ad, deger, baslik=None):
    """Yüzdeleri dilimlerin içinde gösteren halka grafik."""
    grafik = px.pie(tablo, names=ad, values=deger, hole=0.4, title=baslik, height=500)
    grafik.update_traces(textinfo="percent", textposition="inside")
    return grafik.update_layout(legend=dict(orientation="h", y=-0.1)) 


def ilk_n_filtre(df, kolon, yer, varsayilan, key):
    """İlk n seçimi yapar; veriyi kolondaki en yüksek satışlı ilk n değere filtreler."""
    adet = df[kolon].nunique()
    n = n_sec(yer, "Satış tutarına göre ilk n değer", adet, varsayilan or adet, key=key)
    ilk_n = df.groupby(kolon)["Satis"].sum().nlargest(n).index
    return df[df[kolon].isin(ilk_n)]


def ilk_n_secimi(df, varsayilan=10):
    """Kolon ve ilk n seçimi yapar (varsayilan=None ise tüm değerlerle başlar)."""
    k1, k2 = st.columns(2)
    kolon = k1.selectbox("Analiz İçeriği", ["Description", "Category", "City"])
    d = ilk_n_filtre(df, kolon, k2, varsayilan, key=f"ilk_n_{kolon}_{varsayilan}")

    if varsayilan is None:
        k2.caption("Varsayılan olarak en yüksek n değeri seçilmiştir.")
    return kolon, d


def tahmin_et(seri, yontem, n):
    """Aylık seriyi n ay ileriye tahmin eder."""
    y = seri.values.astype(float)

    if yontem == "Hareketli Ortalama":          # pencere = 3 ay
        uyum = seri.rolling(3).mean().shift(1).values
        degerler = list(y)
        for _ in range(n):
            degerler.append(np.mean(degerler[-3:]))
        tahmin = degerler[-n:]
    elif yontem == "Üstel Düzeltme":            # alpha otomatik
        model = SimpleExpSmoothing(y, initialization_method="estimated").fit()
        uyum, tahmin = model.fittedvalues, model.forecast(n)
    else:                                       # ARIMA(1,1,1)
        model = ARIMA(y, order=(1, 1, 1)).fit()
        uyum, tahmin = np.array(model.fittedvalues), model.forecast(n)
        uyum[0] = np.nan                        # fark alındığı için ilk ayın uyum değeri yoktur
    gelecek = pd.date_range(seri.index[-1], periods=n + 1, freq="MS")[1:]
    return pd.Series(uyum, seri.index), pd.Series(np.clip(tahmin, 0, None), gelecek)


def tablo_tahmini(temel, yontem, n):
    """Temel metrikleri tek tek tahmin eder, oranları bu tahminlerden hesaplar."""
    uyum, tahmin = {}, {}

    for ad in temel.columns:
        uyum[ad], tahmin[ad] = tahmin_et(temel[ad], yontem, n)
    return oran_ekle(pd.DataFrame(uyum)), oran_ekle(pd.DataFrame(tahmin))


def hata_olculeri(gercek, tahmin, olcek):
    """Her metrik için 5 hata ölçüsünü hesaplar."""
    hata = (gercek - tahmin).abs()
    return pd.DataFrame({
        "MAE": hata.mean(),
        "RMSE": (hata ** 2).mean() ** 0.5,
        "MAPE (%)": 100 * (hata / gercek.where(gercek != 0)).mean(),
        "sMAPE (%)": 100 * (2 * hata / (gercek.abs() + tahmin.abs())).mean(),
        "MASE": hata.mean() / olcek,
    }).replace([np.inf, -np.inf], np.nan)


ORANLAR = ["Müşteri Başına Satış", "İşlem Başına Satış", "İşlem Başına Ürün Adedi"]

@st.cache_data
def metrik_tahmini(df, ilk_ay, son_ay, yontem, n):
    """Gelecek n ayı tahmin eder; ayrıca son %20 ayı test ayırarak hata ölçülerini hesaplar."""
    aylar = pd.date_range(ilk_ay, son_ay, freq="MS")
    gecmis = metrikler(df, "Ay").reindex(aylar).fillna(0)
    temel = gecmis.drop(columns=ORANLAR)
    tahmin = tablo_tahmini(temel, yontem, n)[1]

    k = max(1, round(len(gecmis) * 0.2))               
    egitim, test = gecmis[:-k], gecmis[-k:]
    uyum, test_tahmin = tablo_tahmini(temel[:-k], yontem, k)
    olcek = egitim.diff().abs().mean()
    hata = pd.concat({"Train": hata_olculeri(egitim, uyum, olcek),
                      "Test": hata_olculeri(test, test_tahmin, olcek)}, axis=1)
    return gecmis, tahmin, test_tahmin, hata


# Sayfalar
def isi_haritasi(tablo, satir, sutun, deger):
    """Satır x sütun kırılımında ısı haritası çizer."""
    pivot = tablo.pivot_table(index=satir, columns=sutun, values=deger, aggfunc="sum")
    return px.imshow(pivot, aspect="auto", color_continuous_scale="Blues")


def ilk_n_satis(df, kolon, yer):
    """Kullanıcının seçtiği n'e göre kolondaki en yüksek satışlı ilk n değeri döndürür."""
    n = n_sec(yer, "En Çok Satış Tutarına Sahip İlk N Değer", df[kolon].nunique(), 10, key=f"ozet_{kolon}")
    return pay_ekle(df.groupby(kolon)["Satis"].sum().nlargest(n).reset_index(), "Satis")


def yonetici_ozeti(df):
    ilk, son = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
    aralik = st.date_input("Dönem aralığı", (ilk, son), min_value=ilk, max_value=son, format="DD.MM.YYYY")

    if len(aralik) != 2:                       
        st.info("Lütfen bitiş tarihini de seçin.")
        return
    
    df = df[df["InvoiceDate"].dt.date.between(*aralik)]

    if df.empty:
        st.warning("Seçilen dönemde satış kaydı yok.")
        return
    
    st.caption(f"{aralik[0]:%d.%m.%Y} - {aralik[1]:%d.%m.%Y} arası gösteriliyor ({len(df):,} satır).")
    aylik = metrikler(df, "Ay")
    kartlar(toplam_metrikler(df))
    genel, trendler, tablo = st.tabs(["Genel Bakış", "Tüm Metrik Trendleri", "Aylık Tablo"])

    with genel:
        st.plotly_chart(px.line(aylik, y=["Satış Tutarı", "İndirim Tutarı"], markers=True,
                                title="Aylık Satış ve İndirim Trendi"))
        sol, sag = st.columns(2)
        sol.markdown("**Kategori Payı**")
        sol.plotly_chart(pasta(ilk_n_satis(df, "Category", sol), "Category", "Satis")
                         .update_layout(margin_t=10))
        sag.markdown("**En Çok Satış Yapılan İller**")
        sag.plotly_chart(px.bar(ilk_n_satis(df, "City", sag).sort_values("Satis"), x="Satis", y="City",
                                orientation="h", text="Pay", height=500).update_layout(margin_t=10))
        st.markdown("**En Çok Satan Ürünler**")
        st.plotly_chart(px.treemap(ilk_n_satis(df, "Description", st), path=["Description"], values="Satis")
                        .update_traces(textinfo="label+percent root").update_layout(margin_t=10))
    with trendler:
        tum_grafikler({ad: aylik[ad] for ad in aylik.columns})
    with tablo:
        st.dataframe(aylik.round(2))


def dinamik_gruplama(df):
    donem = "Ay"                              
    kolon, d = ilk_n_secimi(df, varsayilan=None)

    st.subheader("Seçilen Değerlerin Birleşik Özeti")
    kartlar(toplam_metrikler(d))
    birlesik = metrikler(d, donem)

    st.subheader("Tüm Metriklerin Aylık Trendi (Birleşik)")
    tum_grafikler({ad: birlesik[ad] for ad in birlesik.columns})

    st.subheader("Metrik Bazında İnceleme")
    k1, k2 = st.columns(2)
    metrik = k1.selectbox("Metrik", birlesik.columns)
    d = ilk_n_filtre(df, kolon, k2, 10, key=f"detay_n_{kolon}") 
    detay = metrikler(d, [donem, kolon]).reset_index()
    detay[donem] = detay[donem].dt.strftime("%Y-%m")  
    detay = pay_ekle(detay, metrik, donem)         

    st.plotly_chart(px.line(detay, x=donem, y=metrik, color=kolon, markers=True, title="Aylık Trend"))
    sol, sag = st.columns(2)
    sol.plotly_chart(px.bar(detay, x=donem, y=metrik, color=kolon, text="Pay",
                            title="Ay Bazında Yığılmış Görünüm"))
    sag.plotly_chart(isi_haritasi(detay, kolon, donem, metrik).update_layout(title="Isı Haritası"))

    st.subheader("Metrik Tablosu")
    st.dataframe(detay.drop(columns="Pay").round(2))


def tahminler(df):
    k1, k2 = st.columns(2)
    n = n_sec(k1, "Tahmin edilecek ay sayısı (n)", 24, 1)
    yontem = k2.selectbox("Tahmin Yöntemi", ["Hareketli Ortalama", "Üstel Düzeltme", "ARIMA"])
    kolon, d = ilk_n_secimi(df)
    ilk_ay, son_ay = df["Ay"].min(), df["Ay"].max()

    gecmis, tahmin, test_tahmin, hata = metrik_tahmini(d, ilk_ay, son_ay, yontem, n)
    st.subheader(f"Seçilen Değerlerin Birleşik Tahmini ({n}. ay)")
    kartlar(tahmin.iloc[-1])

    st.subheader("Tüm Metriklerin Tahmini (Birleşik)")
    tum_grafikler({ad: pd.DataFrame({"Gerçekleşen": gecmis[ad], "Tahmin": tahmin[ad]})
                   for ad in tahmin.columns})

    st.subheader("Metrik Bazında Tahmin")
    metrik = st.selectbox("Metrik", tahmin.columns)
    parcalar, deger_hatalari = [], {}
    for deger in d[kolon].unique():
        g, t, _, h = metrik_tahmini(d[d[kolon] == deger], ilk_ay, son_ay, yontem, n)
        deger_hatalari[deger] = h.loc[metrik]
        parcalar.append(pd.concat([g.assign(Tip="Gerçekleşen"), t.assign(Tip="Tahmin")])
                        .assign(**{kolon: deger}))
    detay = pd.concat(parcalar).rename_axis("Ay").reset_index()
    st.plotly_chart(px.line(detay, x="Ay", y=metrik, color=kolon, line_dash="Tip", markers=True,
                            title="Gerçekleşen ve Tahmin"))

    sadece_tahmin = detay[detay["Tip"] == "Tahmin"].drop(columns="Tip")
    sadece_tahmin["Ay"] = sadece_tahmin["Ay"].dt.strftime("%Y-%m")
    sadece_tahmin = pay_ekle(sadece_tahmin, metrik, "Ay")
    toplam = pay_ekle(sadece_tahmin.groupby(kolon, as_index=False)[metrik].sum(), metrik)
    sol, sag = st.columns(2)
    sol.plotly_chart(px.bar(sadece_tahmin, x="Ay", y=metrik, color=kolon, barmode="group",
                            text="Pay", title="Tahmin Edilen Aylar"))
    sag.plotly_chart(pasta(toplam, kolon, metrik, "Tahmin Dönemi Pay Dağılımı"))
    st.subheader("Tahmin Hata Payı (Train / Test)")
    st.caption(f"Geçmiş {len(gecmis)} ayın ilk {len(gecmis) - len(test_tahmin)} ayı ile model kuruldu (train), "
               f"son {len(test_tahmin)} ayı (%20) tahmin edilip gerçekleşenle karşılaştırıldı (test). "
               "MAE: ortalama mutlak hata (metriğin biriminde) · RMSE: büyük sapmaları daha ağır cezalandırır · "
               "MAPE: ortalama mutlak yüzde hata · sMAPE: simetrik yüzde hata (0'a yakın değerlerde daha güvenilir) · "
               "MASE: 1'den küçükse model, 'bir önceki ayın aynısı' diyen naif tahminden iyidir.")
    train, test = hata.loc[metrik, "Train"], hata.loc[metrik, "Test"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Train MAPE", f"%{train['MAPE (%)']:.1f}")
    k2.metric("Test MAPE", f"%{test['MAPE (%)']:.1f}")
    k3.metric("Test MASE", f"{test['MASE']:.2f}")
    st.info(f"**{metrik}** için {yontem} tahminleri test döneminde gerçekleşenden ortalama "
            f"%{test['MAPE (%)']:.1f} (±{test['MAE']:,.2f}) saptı. "
            + ("Test hatası train hatasının 1,5 katından fazla: model görmediği dönemde belirgin şekilde "
               "zayıflıyor, tahminlere temkinli yaklaşılmalı. " if test["MAE"] > 1.5 * train["MAE"]
               else "Test hatası train hatasına yakın: model görmediği dönemde de benzer başarıyı koruyor. ")
            + ("MASE 1'in altında: model naif tahminden daha iyi." if test["MASE"] < 1
               else "MASE 1'in üzerinde: model naif tahminden daha iyi değil." if test["MASE"] >= 1
               else "Seri sabit olduğu için MASE hesaplanamadı."))

    karsilastirma = pd.DataFrame({"Gerçekleşen": gecmis[metrik], "Test Tahmini": test_tahmin[metrik]})
    st.plotly_chart(px.line(karsilastirma, markers=True, title="Test Döneminde Gerçekleşen ve Tahmin"))
    st.write("**Birleşik seride tüm metriklerin hata ölçüleri**")
    st.dataframe(hata.round(2))
    st.write(f"**Değer bazında hata ölçüleri ({metrik})**")
    deger_hata = pd.DataFrame(deger_hatalari).T
    st.dataframe(deger_hata.round(2))
    st.subheader("Tahmin Tablosu")
    st.dataframe(sadece_tahmin.drop(columns="Pay").round(2))


SAYFALAR = {"Yönetici Özeti": yonetici_ozeti,
            "Detay KPI ve Metrikler": dinamik_gruplama,
            "Tahminler": tahminler}
st.title("B2B Performans Analizi")
sayfa = st.segmented_control("Sayfa", list(SAYFALAR), default="Yönetici Özeti",
                             label_visibility="collapsed") or "Yönetici Özeti"
st.header(sayfa)
SAYFALAR[sayfa](imputation_paneli(veri_oku()))
