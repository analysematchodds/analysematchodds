import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

def load_and_prepare_data(csv_path, selected_columns):
    try:
        # CSV'yi oku
        df = pd.read_csv(csv_path)
        
        # Tüm sayısal sütunları belirle
        numeric_columns = [
            'MS1', 'MS0', 'MS2', 'AU2.5 Alt', 'AU2.5 Üst', 
            'KG Var', 'KG Yok', 'IY0.5 Alt', 'IY0.5 Üst',
            'AU1.5 Alt', 'AU1.5 Üst', 'IY1', 'IY0', 'IY2',
            '2Y1', '2Y0', '2Y2', 'Tek', 'Çift',
            'IY/MS 1/1', 'IY/MS 1/0', 'IY/MS 1/2',
            'IY/MS 0/1', 'IY/MS 0/0', 'IY/MS 0/2',
            'IY/MS 2/1', 'IY/MS 2/0', 'IY/MS 2/2'
        ]
        
        # Tüm sayısal sütunları float'a çevir ve yuvarla
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(2).fillna(0)
        
        # Seçilen sütunlar için eksik ve '-' değerlerini filtrele
        for col in selected_columns:
            df = df[df[col] != '-']  # '-' değerlerini kaldır
        
        df.fillna(0, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None

@st.cache_data
def calculate_similarity(df, numeric_columns):
    """
    Maçlar arasındaki benzerlik matrisini hesaplar.
    
    Parametreler:
    - df: Maç verileri DataFrame'i
    - numeric_columns: Benzerlik hesaplanacak sayısal sütunlar
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[numeric_columns])
    
    n_matches = len(df)
    similarity_matrix = np.zeros((n_matches, n_matches))
    
    # Sütunlara göre ağırlıklar tanımla
    weights = {}
    for col in numeric_columns:
        if 'MS' in col:  # Maç Sonucu
            weights[numeric_columns.index(col)] = 2.0
        elif 'KG' in col:  # Karşılıklı Gol
            weights[numeric_columns.index(col)] = 1.8
        elif 'IY/MS' in col:  # İlk Yarı/Maç Sonucu
            weights[numeric_columns.index(col)] = 1.7
        elif 'IY' in col and not 'IY/MS' in col:  # İlk Yarı (IY/MS hariç)
            weights[numeric_columns.index(col)] = 1.6
        elif '2Y' in col:  # İkinci Yarı
            weights[numeric_columns.index(col)] = 1.5
        elif 'AU2.5' in col:  # Alt/Üst 2.5
            weights[numeric_columns.index(col)] = 1.4
        elif 'AU1.5' in col:  # Alt/Üst 1.5
            weights[numeric_columns.index(col)] = 1.3
        elif 'Tek' in col or 'Çift' in col:  # Tek/Çift
            weights[numeric_columns.index(col)] = 1.2
        else:
            weights[numeric_columns.index(col)] = 1.0
    
    weight_array = np.array([weights.get(i, 1.0) for i in range(len(numeric_columns))])
    
    # Her maç çifti için benzerlik hesapla
    for i in range(n_matches):
        for j in range(n_matches):
            if i != j:
                # Seçilen sütunlardaki oranların farkını hesapla
                diff = np.abs(normalized_data[i] - normalized_data[j])
                
                # Ağırlıklı fark hesapla
                weighted_diff = diff * weight_array
                
                # Her bir oran için maksimum kabul edilebilir fark
                max_acceptable_diff = 0.2  # %20'lik farka yükseltildi
                
                # Farkın kabul edilebilir sınırlar içinde olup olmadığını kontrol et
                within_threshold = weighted_diff <= max_acceptable_diff
                
                # En az %70 oranında benzerlik varsa kabul et
                min_similar_features = int(len(numeric_columns) * 0.7)
                if np.sum(within_threshold) >= min_similar_features:
                    # Benzerlik skorunu hesapla
                    similarity = 1 - (np.mean(weighted_diff) / max_acceptable_diff)
                    similarity_matrix[i,j] = similarity
                else:
                    similarity_matrix[i,j] = 0
    
    return similarity_matrix

def find_similar_matches(df, similarity_matrix, selected_idx, n_matches=5):
    """
    Seçilen maça en benzer n maçı bulur.
    
    Parametreler:
    - df: Maç verileri DataFrame'i
    - similarity_matrix: Benzerlik matrisi
    - selected_idx: Seçilen maçın indeksi
    - n_matches: Bulunacak benzer maç sayısı
    """
    similarities = similarity_matrix[selected_idx]
    
    # Başlangıç benzerlik eşiği
    initial_threshold = 0.5
    min_threshold = 0.3
    
    similarity_threshold = initial_threshold
    similar_indices = []
    
    # Tarih bazlı filtreleme için
    selected_date = pd.to_datetime(df.iloc[selected_idx]['Tarih'], format='%d.%m.%Y')
    
    # Yeterli benzer maç bulana kadar eşiği düşür
    while len(similar_indices) < n_matches and similarity_threshold >= min_threshold:
        # Eşik değerini geçen maçları filtrele
        candidate_indices = np.where(similarities >= similarity_threshold)[0]
        
        # Tarih kontrolü yap (seçilen maçtan önceki maçları al)
        filtered_indices = []
        for idx in candidate_indices:
            if idx != selected_idx:
                match_date = pd.to_datetime(df.iloc[idx]['Tarih'], format='%d.%m.%Y')
                if match_date < selected_date:
                    filtered_indices.append(idx)
        
        # Benzerlik skoruna göre sırala
        similar_indices = sorted(filtered_indices, 
                               key=lambda x: similarities[x], 
                               reverse=True)[:n_matches]
        
        # Yeterli maç bulunamadıysa eşiği düşür
        if len(similar_indices) < n_matches:
            similarity_threshold -= 0.1
    
    if len(similar_indices) == 0:
        st.warning("Yeterince benzer maç bulunamadı! Lütfen farklı benzerlik kriterleri seçin.")
        return None
    
    # Seçilen maç ve benzer maçları DataFrame'e ekle
    matches = [df.iloc[selected_idx]]
    similarities_list = [1.0]
    
    for idx in similar_indices:
        matches.append(df.iloc[idx])
        similarities_list.append(similarities[idx])
    
    result_df = pd.concat(matches, axis=1).T.reset_index(drop=True)
    result_df['Similarity'] = similarities_list
    
    return result_df

def main():
    st.title("Benzer Maç Bulucu")
    
    # Sol panel için container
    with st.container():
        st.write("### Benzerlik Kriterleri")
        
        # Her bir kriter için checkbox'ları 2 sütunda göster
        similarity_options = {
            'Maç Sonucu': ['MS1', 'MS0', 'MS2'],
            'Alt/Üst 2.5': ['AU2.5 Alt', 'AU2.5 Üst'],
            'Karşılıklı Gol': ['KG Var', 'KG Yok'],
            'İlk Yarı Alt/Üst 0.5': ['IY0.5 Alt', 'IY0.5 Üst'],
            'Alt/Üst 1.5': ['AU1.5 Alt', 'AU1.5 Üst'],
            'İlk Yarı Sonucu': ['IY1', 'IY0', 'IY2'],
            'İkinci Yarı Sonucu': ['2Y1', '2Y0', '2Y2'],
            'Tek/Çift': ['Tek', 'Çift'],
            'İlk Yarı/Maç Sonucu': [
                'IY/MS 1/1', 'IY/MS 1/0', 'IY/MS 1/2',
                'IY/MS 0/1', 'IY/MS 0/0', 'IY/MS 0/2',
                'IY/MS 2/1', 'IY/MS 2/0', 'IY/MS 2/2'
            ]
        }
        
        selected_criteria = []
        cols = st.columns(2)  # 2 sütunlu layout
        for i, (name, columns) in enumerate(similarity_options.items()):
            if cols[i % 2].checkbox(name, value=(name == 'Maç Sonucu')):
                selected_criteria.extend(columns)
    
    st.write("---")  # Ayırıcı çizgi
    
    if not selected_criteria:
        st.warning("En az bir benzerlik kriteri seçmelisiniz!")
        return
    
    # Lig Seçimi
    st.header("Lig Seçimi")
    leagues = {
        "Süper Lig": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/matchodds.csv",
        "Premier League": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/premierleague.csv",
        "La Liga": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/laliga.csv",
        "Serie A": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/seriea.csv",
        "Bundesliga": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/bundesliga.csv",
        "Ligue 1": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/ligue1.csv"
    }

    selected_league = st.selectbox("Lig Seçin", options=list(leagues.keys()))
    csv_path = leagues[selected_league]
    
    # CSV dosyasını seçilen kriterlere göre oku
    df = load_and_prepare_data(csv_path, selected_criteria)
    
    if df is None:
        return
    
    df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%d.%m.%Y')
    df['Saat'] = pd.to_datetime(df['Saat'], format='%H:%M:%S').dt.strftime('%H:%M')

    # Benzerlik matrisini hesapla
    similarity_matrix = calculate_similarity(df, selected_criteria)
    
    st.write("---")  # Ayırıcı çizgi
    
    # En son haftayı bul ve o haftanın maçlarını filtrele
    latest_week = df['Hafta'].max()
    latest_week_matches = df[df['Hafta'] == latest_week]

    if not latest_week_matches.empty:
        st.write("Bu haftanın maçları:")

        # Tıklanabilir maç listesi oluştur
        week_matches = []
        for idx, row in latest_week_matches.iterrows():
            match_str = f"{row['Ev Sahibi']} vs {row['Deplasman']} ({row['Tarih']} - {row['Saat']})"
            week_matches.append((df.index.get_loc(idx), match_str))

        selected_week_match = st.selectbox(
            "Haftanın maçlarından birini seçin:",
            options=week_matches,
            format_func=lambda x: x[1]
        )

        if st.button("Seçilen Maçın Benzerlerini Bul"):
            selected_idx = selected_week_match[0]
            similarity_matrix = calculate_similarity(df, selected_criteria)
            result_df = find_similar_matches(df, similarity_matrix, selected_idx)
            
            if result_df is not None:
                # Format sözlüğünü oluştur
                format_dict = {
                    'MS1': '{:.2f}', 'MS0': '{:.2f}', 'MS2': '{:.2f}',
                    'AU2.5 Alt': '{:.2f}', 'AU2.5 Üst': '{:.2f}',
                    'KG Var': '{:.2f}', 'KG Yok': '{:.2f}',
                    'IY0.5 Alt': '{:.2f}', 'IY0.5 Üst': '{:.2f}',
                    'AU1.5 Alt': '{:.2f}', 'AU1.5 Üst': '{:.2f}',
                    'IY1': '{:.2f}', 'IY0': '{:.2f}', 'IY2': '{:.2f}',
                    '2Y1': '{:.2f}', '2Y0': '{:.2f}', '2Y2': '{:.2f}',
                    'Tek': '{:.2f}', 'Çift': '{:.2f}',
                    'IY/MS 1/1': '{:.2f}', 'IY/MS 1/0': '{:.2f}', 'IY/MS 1/2': '{:.2f}',
                    'IY/MS 0/1': '{:.2f}', 'IY/MS 0/0': '{:.2f}', 'IY/MS 0/2': '{:.2f}',
                    'IY/MS 2/1': '{:.2f}', 'IY/MS 2/0': '{:.2f}', 'IY/MS 2/2': '{:.2f}',
                    'Similarity': '{:.2%}'
                }

                # Temel bilgi sütunları + seçili kriterler
                base_columns = ['Tarih', 'Saat', 'Lig', 'MBS', 'Ev Sahibi', 'Skor', 'Deplasman', 'İY']
                
                # Seçilen maçı göster
                st.write("### Seçilen Maç")
                selected_columns = base_columns + selected_criteria
                st.dataframe(
                    result_df.iloc[[0]][selected_columns].style.format(format_dict)
                )

                # Benzer maçları göster
                st.write("### En Benzer 5 Maç")
                display_columns = ['Similarity'] + base_columns + selected_criteria
                
                # Format sözlüğünü seçili kriterler için güncelle
                selected_format_dict = {col: '{:.2f}' for col in selected_criteria}
                selected_format_dict['Similarity'] = '{:.2%}'
                
                st.dataframe(
                    result_df.iloc[1:][display_columns].style.format(selected_format_dict)
                )

                # Analiz Bölümü
                with st.container():
                    st.write("### Benzer Maçların Analizi")
                    
                    # Goller listesini başta hazırla
                    goller = []
                    sonuclar = []
                    for _, row in result_df.iloc[1:].iterrows():
                        if row['Skor'] != '-':
                            ev_gol, dep_gol = map(int, row['Skor'].split('-'))
                            toplam_gol = ev_gol + dep_gol
                            goller.append(toplam_gol)
                            
                            # Sonuçları da burada hesapla
                            if ev_gol > dep_gol:
                                sonuclar.append("Ev Sahibi Kazandı")
                            elif ev_gol < dep_gol:
                                sonuclar.append("Deplasman Kazandı")
                            else:
                                sonuclar.append("Berabere")
                    
                    sonuc_dagilimi = pd.Series(sonuclar).value_counts()
                    
                    # Metrik kartları
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        ortalama_gol = sum(goller) / len(goller) if goller else 0
                        st.metric(
                            "Ortalama Gol",
                            f"{ortalama_gol:.1f}",
                            help="Benzer maçlardaki ortalama gol sayısı"
                        )
                    
                    with col2:
                        # İlk yarı gol ortalaması
                        iy_goller = []
                        for _, row in result_df.iloc[1:].iterrows():
                            if row['İY'] != '-':
                                iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                iy_goller.append(iy_ev + iy_dep)
                        
                        iy_ort = sum(iy_goller) / len(iy_goller) if iy_goller else 0
                        st.metric(
                            "İlk Yarı Gol Ortalaması",
                            f"{iy_ort:.1f}",
                            help="Benzer maçlardaki ilk yarı ortalama gol sayısı"
                        )
                    
                    with col3:
                        # Maç başına gol dağılımı analizi
                        mac_gol_dagilimi = {
                            "0-1 Gol": 0,
                            "2-3 Gol": 0,
                            "4+ Gol": 0
                        }
                        
                        for gol in goller:
                            if gol <= 1:
                                mac_gol_dagilimi["0-1 Gol"] += 1
                            elif gol <= 3:
                                mac_gol_dagilimi["2-3 Gol"] += 1
                            else:
                                mac_gol_dagilimi["4+ Gol"] += 1
                        
                        en_yaygin_aralik = max(mac_gol_dagilimi.items(), key=lambda x: x[1])[0]
                        en_yaygin_oran = (mac_gol_dagilimi[en_yaygin_aralik] / len(goller) * 100)
                        
                        st.metric(
                            "En Yaygın Gol Aralığı",
                            en_yaygin_aralik,
                            f"%{en_yaygin_oran:.0f}",
                            help="Benzer maçlarda en sık görülen gol aralığı ve yüzdesi"
                        )

                    # Grafikler için sekmeler
                    tab1, tab2, tab3 = st.tabs(["Detaylı İstatistikler", "Gol Analizi", "İY/MS Analizi"])
                    
                    with tab1:
                        # Detaylı istatistikler
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("##### Gol İstatistikleri")
                            st.write(f"- En çok gol: {max(goller) if goller else '-'}")
                            st.write(f"- En az gol: {min(goller) if goller else '-'}")
                            
                            # 2.5 Üstü analizi
                            ustu_mac = len([g for g in goller if g > 2])
                            toplam_mac = len(goller)
                            ustu_oran = (ustu_mac / toplam_mac * 100) if toplam_mac > 0 else 0
                            st.write(f"- 2.5 Üstü Maç Oranı: {ustu_oran:.1f}% ({ustu_mac}/{toplam_mac})")
                            
                            # İlk Yarı 1.5 Üstü analizi
                            iy_15_ustu = 0
                            toplam_mac = 0
                            for _, row in result_df.iloc[1:].iterrows():
                                if row['İY'] != '-':
                                    iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                    if (iy_ev + iy_dep) > 1:
                                        iy_15_ustu += 1
                                    toplam_mac += 1
                            
                            iy_15_oran = (iy_15_ustu / toplam_mac * 100) if toplam_mac > 0 else 0
                            st.write(f"- İY 1.5 Üstü Oranı: {iy_15_oran:.1f}% ({iy_15_ustu}/{toplam_mac})")
                            
                            # Karşılıklı Gol analizi
                            kg_var = 0
                            toplam_mac = 0
                            for _, row in result_df.iloc[1:].iterrows():
                                if row['Skor'] != '-':
                                    ev_gol, dep_gol = map(int, row['Skor'].split('-'))
                                    if ev_gol > 0 and dep_gol > 0:
                                        kg_var += 1
                                    toplam_mac += 1
                            
                            kg_oran = (kg_var / toplam_mac * 100) if toplam_mac > 0 else 0
                            st.write(f"- Karşılıklı Gol Oranı: {kg_oran:.1f}% ({kg_var}/{toplam_mac})")
                        
                        with col2:
                            st.write("##### Sonuç İstatistikleri")
                            sonuc_dagilimi = pd.Series(sonuclar).value_counts()
                            for sonuc, count in sonuc_dagilimi.items():
                                st.write(f"- {sonuc}: {count} maç ({count/len(sonuclar)*100:.1f}%)")
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Mevcut gol dağılımı grafiği
                            gol_dagilimi = pd.Series(goller).value_counts().sort_index()
                            fig = px.bar(
                                x=gol_dagilimi.index,
                                y=gol_dagilimi.values,
                                title="Toplam Gol Dağılımı",
                                labels={'x': 'Toplam Gol', 'y': 'Maç Sayısı'},
                                color_discrete_sequence=['#FF9999']
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # İlk Yarı/İkinci Yarı gol karşılaştırması
                            iy_goller = []
                            iy2_goller = []
                            for _, row in result_df.iloc[1:].iterrows():
                                if row['İY'] != '-' and row['Skor'] != '-':
                                    iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                    ms_ev, ms_dep = map(int, row['Skor'].split('-'))
                                    iy_goller.append(iy_ev + iy_dep)
                                    iy2_goller.append((ms_ev + ms_dep) - (iy_ev + iy_dep))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Box(y=iy_goller, name="İlk Yarı"))
                            fig.add_trace(go.Box(y=iy2_goller, name="İkinci Yarı"))
                            fig.update_layout(title="Yarı Bazında Gol Dağılımı")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Mevcut İY/MS analizi grafiği (geliştirilmiş)
                        iy_ms_analiz = []
                        for _, row in result_df.iloc[1:].iterrows():
                            if row['İY'] != '-' and row['Skor'] != '-':
                                iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                ms_ev, ms_dep = map(int, row['Skor'].split('-'))
                                
                                # İlk yarı sonucu
                                if iy_ev > iy_dep: iy_sonuc = "1"
                                elif iy_ev < iy_dep: iy_sonuc = "2"
                                else: iy_sonuc = "0"
                                
                                # Maç sonucu
                                if ms_ev > ms_dep: ms_sonuc = "1"
                                elif ms_ev < ms_dep: ms_sonuc = "2"
                                else: ms_sonuc = "0"
                                
                                iy_ms_analiz.append(f"{iy_sonuc}/{ms_sonuc}")

                        iy_ms_dagilimi = pd.Series(iy_ms_analiz).value_counts()
                        fig = px.bar(
                            x=iy_ms_dagilimi.index,
                            y=iy_ms_dagilimi.values,
                            title="İY/MS Kombinasyonları",
                            labels={'x': 'İY/MS', 'y': 'Maç Sayısı'},
                            color=iy_ms_dagilimi.values,
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
