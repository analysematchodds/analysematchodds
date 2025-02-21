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
        
        # Tarih ve saat formatlarını düzelt
        df['Tarih'] = pd.to_datetime(df['Tarih'], format='%Y-%m-%d').dt.strftime('%d.%m.%Y')
        df['Saat'] = pd.to_datetime(df['Saat'], format='%H:%M:%S').dt.strftime('%H:%M')
        
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
    
    # Vektörizasyon ile performans iyileştirmesi
    normalized_data_matrix = normalized_data[:, np.newaxis, :]
    diff_matrix = np.abs(normalized_data_matrix - normalized_data)
    weighted_diff_matrix = diff_matrix * weight_array
    
    within_threshold = weighted_diff_matrix <= 0.25  # %20'den %25'e yükseltildi
    similar_features_count = np.sum(within_threshold, axis=2)
    
    min_similar_features = int(len(numeric_columns) * 0.6)
    valid_similarities = similar_features_count >= min_similar_features
    
    similarity_matrix = np.where(
        valid_similarities,
        1 - (np.mean(weighted_diff_matrix, axis=2) / 0.25),
        0
    )
    
    return similarity_matrix

def find_similar_matches(df, similarity_matrix, selected_idx, n_matches=5, 
                        min_similarity=0.5):
    """
    Seçilen maça en benzer n maçı bulur.
    
    Parametreler:
    - df: Maç verileri DataFrame'i
    - similarity_matrix: Benzerlik matrisi
    - selected_idx: Seçilen maçın indeksi
    - n_matches: Bulunacak benzer maç sayısı
    - min_similarity: Minimum benzerlik oranı
    """
    similarities = similarity_matrix[selected_idx]
    
    # Tarih formatını düzelt
    selected_date = pd.to_datetime(df.iloc[selected_idx]['Tarih'], format='%d.%m.%Y', dayfirst=True)
    
    # Tarih kontrolü yap (seçilen maçtan önceki maçları al)
    filtered_indices = []
    for idx, similarity in enumerate(similarities):
        if idx != selected_idx and similarity >= min_similarity:
            match_date = pd.to_datetime(df.iloc[idx]['Tarih'], format='%d.%m.%Y', dayfirst=True)
            
            # Filtreleme kriteri
            date_ok = match_date < selected_date
            
            if date_ok:
                filtered_indices.append(idx)
    
    # Benzerlik skoruna göre sırala ve en iyi n tanesini al
    similar_indices = sorted(filtered_indices, 
                           key=lambda x: similarities[x], 
                           reverse=True)[:n_matches]
    
    if len(similar_indices) == 0:
        st.warning("Seçilen kriterlere uygun benzer maç bulunamadı! Lütfen filtreleri gevşetin.")
        return None
    
    # Seçilen maç ve benzer maçları DataFrame'e ekle
    matches = [df.iloc[selected_idx]]
    similarities_list = [1.0]
    
    for idx in similar_indices:
        matches.append(df.iloc[idx])
        similarities_list.append(similarities[idx])
    
    result_df = pd.concat(matches, axis=1).T.reset_index(drop=True)
    result_df['Similarity'] = similarities_list
    
    # Tarih ve saat formatlarını düzelt
    result_df['Tarih'] = pd.to_datetime(result_df['Tarih'], format='%d.%m.%Y', dayfirst=True).dt.strftime('%d.%m.%Y')
    result_df['Saat'] = pd.to_datetime(result_df['Saat'], format='%H:%M').dt.strftime('%H:%M')
    
    return result_df

def analyze_team_performance(df, team_name, last_n_matches=5):
    """
    Takımın son n maçtaki performans analizini yapar
    """
    team_matches = df[
        ((df['Ev Sahibi'] == team_name) | (df['Deplasman'] == team_name))
    ].copy()
    
    team_matches['Tarih'] = pd.to_datetime(team_matches['Tarih'], format='%d.%m.%Y')
    team_matches = team_matches.sort_values('Tarih', ascending=False).head(last_n_matches)
    
    metrics = {
        'Galibiyet': 0,
        'Beraberlik': 0,
        'Mağlubiyet': 0,
        'Atılan Gol': 0,
        'Yenilen Gol': 0
    }
    
    for _, match in team_matches.iterrows():
        ev_gol, dep_gol = map(int, match['Skor'].split('-'))
        
        if match['Ev Sahibi'] == team_name:
            metrics['Atılan Gol'] += ev_gol
            metrics['Yenilen Gol'] += dep_gol
            if ev_gol > dep_gol:
                metrics['Galibiyet'] += 1
            elif ev_gol < dep_gol:
                metrics['Mağlubiyet'] += 1
            else:
                metrics['Beraberlik'] += 1
        else:
            metrics['Atılan Gol'] += dep_gol
            metrics['Yenilen Gol'] += ev_gol
            if dep_gol > ev_gol:
                metrics['Galibiyet'] += 1
            elif dep_gol < ev_gol:
                metrics['Mağlubiyet'] += 1
            else:
                metrics['Beraberlik'] += 1
    
    return metrics

def validate_data(df):
    """
    Veri setinin doğruluğunu ve güvenliğini kontrol eder
    """
    required_columns = ['Tarih', 'Saat', 'Ev Sahibi', 'Deplasman', 'Skor', 'İY']
    
    # Gerekli sütunların varlığını kontrol et
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Eksik sütunlar bulundu")
    
    # Tarih formatı kontrolü
    try:
        pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
    except:
        raise ValueError("Geçersiz tarih formatı")
    
    # Skor formatı kontrolü
    if not df['Skor'].str.match(r'^\d+-\d+$').all():
        raise ValueError("Geçersiz skor formatı")
    
    return True

def main():
    st.title("Benzer Maç Bulucu")
    
    # Lig Seçimi
    st.header("Lig Seçimi")
    leagues = {
        "Süper Lig": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/matchodds.csv",
        "TFF 1. Lig": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/turkiye1.csv",
        "Premier League": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/premierleague.csv",
        "La Liga": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/laliga.csv",
        "Serie A": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/seriea.csv",
        "Bundesliga": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/bundesliga.csv",
        "Ligue 1": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/ligue1.csv",
        "Şampiyonlar Ligi": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/championsleague.csv",
        "Avrupa Ligi": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/europeleague.csv",
        "Konferans Ligi": "https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/conferenceleague.csv"
    }

    selected_league = st.selectbox("Lig Seçin", options=list(leagues.keys()))
    csv_path = leagues[selected_league]
    
    # CSV dosyasını oku
    df = pd.read_csv(csv_path)
    
    if df is not None:
        # Tarih ve saat formatlarını düzelt
        df['Tarih'] = pd.to_datetime(df['Tarih'], format='%Y-%m-%d').dt.strftime('%d.%m.%Y')
        df['Saat'] = pd.to_datetime(df['Saat'], format='%H:%M:%S').dt.strftime('%H:%M')
        
        # En son haftayı bul ve o haftanın maçlarını filtrele
        latest_week = df['Hafta'].max()
        latest_week_matches = df[df['Hafta'] == latest_week]

        if not latest_week_matches.empty:
            st.write("Bu haftanın maçları:")

            # Tıklanabilir maç listesi oluştur
            week_matches = []
            for idx, row in latest_week_matches.iterrows():
                match_str = f"{row['Ev Sahibi']} vs {row['Deplasman']} ({row['Tarih']} - {row['Saat']})"
                week_matches.append((df.index.get_loc(idx), match_str))  # Tuple olarak bırakıyoruz

            selected_week_match = st.selectbox(
                "Haftanın maçlarından birini seçin:",
                options=week_matches,
                format_func=lambda x: x[1]  # İkinci elemanı (match_str) göster
            )

            # Seçilen maç metnini kopyalanabilir şekilde göster
            if selected_week_match:
                st.code(selected_week_match[1], language=None)  # Tuple'ın ikinci elemanını kullan

            # Daha sonraki işlemler için seçilen maçın index'ini kullan
            selected_idx = selected_week_match[0]  # Tuple'ın ilk elemanını kullan

            # Filtreleme seçenekleri
            st.write("### 🔍 Filtreleme Seçenekleri")
            
            # Benzerlik oranı filtresi
            min_similarity = st.slider(
                "Minimum Benzerlik Oranı",
                min_value=0.30,
                max_value=0.90,
                value=0.30,
                step=0.05,
                help="Seçilen maça benzerlik oranı bu değerin altında olan maçlar gösterilmeyecek"
            )
            
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

            if not selected_criteria:
                st.warning("En az bir benzerlik kriteri seçmelisiniz!")
                return

            if st.button("Seçilen Maçın Benzerlerini Bul"):
                # CSV dosyasını seçilen kriterlere göre oku
                df = load_and_prepare_data(csv_path, selected_criteria)
                
                if df is not None:
                    selected_idx = selected_idx
                    similarity_matrix = calculate_similarity(df, selected_criteria)
                    
                    result_df = find_similar_matches(
                        df, 
                        similarity_matrix, 
                        selected_idx,
                        min_similarity=min_similarity
                    )
                    
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
                            result_df.iloc[[0]][selected_columns].style
                            .format(format_dict)
                        )

                        # Benzer maçları göster
                        st.write("### En Benzer 5 Maç")
                        display_columns = ['Similarity'] + base_columns + selected_criteria
                        
                        # Format sözlüğünü seçili kriterler için güncelle
                        selected_format_dict = {col: '{:.2f}' for col in selected_criteria}
                        selected_format_dict['Similarity'] = '{:.2%}'
                        
                        st.dataframe(
                            result_df.iloc[1:][display_columns].style
                            .format(selected_format_dict)
                        )

                        # Tüm oranları içeren yeni tablo
                        st.write("### Tüm Oranlar")
                        
                        # Tüm oran sütunlarını içeren liste
                        all_odds_columns = [
                            'MS1', 'MS0', 'MS2', 'AU2.5 Alt', 'AU2.5 Üst',
                            'KG Var', 'KG Yok', 'IY0.5 Alt', 'IY0.5 Üst',
                            'AU1.5 Alt', 'AU1.5 Üst', 'IY1', 'IY0', 'IY2',
                            '2Y1', '2Y0', '2Y2', 'Tek', 'Çift',
                            'IY/MS 1/1', 'IY/MS 1/0', 'IY/MS 1/2',
                            'IY/MS 0/1', 'IY/MS 0/0', 'IY/MS 0/2',
                            'IY/MS 2/1', 'IY/MS 2/0', 'IY/MS 2/2'
                        ]
                        
                        # Temel bilgi sütunları + tüm oranlar
                        full_display_columns = ['Similarity'] + base_columns + all_odds_columns
                        
                        # Format sözlüğünü tüm oranlar için güncelle
                        full_format_dict = {col: '{:.2f}' for col in all_odds_columns}
                        full_format_dict['Similarity'] = '{:.2%}'
                        
                        # İlk satır için stil fonksiyonu
                        def highlight_first_row(x):
                            return ['background-color: #272727' if i == 0 else '' for i in range(len(x))]
                        
                        st.dataframe(
                            result_df[full_display_columns].style
                            .format(full_format_dict)
                            .apply(highlight_first_row, axis=0)
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
                            
                            # Metrik kartları - İlk row
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
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
                            
                            with col2:
                                # İlk yarı gol dağılımı analizi
                                iy_gol_dagilimi = {
                                    "0 Gol": 0,
                                    "1 Gol": 0,
                                    "2+ Gol": 0
                                }
                                
                                for _, row in result_df.iloc[1:].iterrows():
                                    if row['İY'] != '-':
                                        iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                        iy_toplam = iy_ev + iy_dep
                                        
                                        if iy_toplam == 0:
                                            iy_gol_dagilimi["0 Gol"] += 1
                                        elif iy_toplam == 1:
                                            iy_gol_dagilimi["1 Gol"] += 1
                                        else:
                                            iy_gol_dagilimi["2+ Gol"] += 1
                                
                                en_yaygin_iy = max(iy_gol_dagilimi.items(), key=lambda x: x[1])[0]
                                en_yaygin_iy_oran = (iy_gol_dagilimi[en_yaygin_iy] / sum(iy_gol_dagilimi.values()) * 100)
                                
                                st.metric(
                                    "İlk Yarı En Yaygın Gol",
                                    en_yaygin_iy,
                                    f"%{en_yaygin_iy_oran:.0f}",
                                    help="Benzer maçlarda ilk yarıda en sık görülen gol sayısı ve yüzdesi"
                                )
                            
                            with col3:
                                st.write("##### İlk Yarı Gol Dağılımı")
                                # İlk yarı gol dağılımını tablo olarak göster
                                for gol_sayisi, mac_sayisi in iy_gol_dagilimi.items():
                                    toplam = sum(iy_gol_dagilimi.values())
                                    yuzde = (mac_sayisi / toplam * 100) if toplam > 0 else 0
                                    st.write(f"**{gol_sayisi}:** {mac_sayisi} maç ({yuzde:.0f}%)")

                            # Taraf analizi için yeni row
                            st.write("")  # Boşluk ekle
                            col4, col5, col6 = st.columns(3)
                            
                            with col4:
                                # Maç sonucu analizi
                                mac_sonuclari = []
                                for _, row in result_df.iloc[1:].iterrows():
                                    if row['Skor'] != '-':
                                        ev_gol, dep_gol = map(int, row['Skor'].split('-'))
                                        if ev_gol > dep_gol:
                                            mac_sonuclari.append("Ev Sahibi")
                                        elif ev_gol < dep_gol:
                                            mac_sonuclari.append("Deplasman")
                                        else:
                                            mac_sonuclari.append("Beraberlik")
                                
                                sonuc_dagilimi = pd.Series(mac_sonuclari).value_counts()
                                en_yaygin_sonuc = sonuc_dagilimi.index[0] if not sonuc_dagilimi.empty else "-"
                                sonuc_oran = (sonuc_dagilimi.iloc[0] / len(mac_sonuclari) * 100) if not sonuc_dagilimi.empty else 0
                                
                                st.metric(
                                    "En Yaygın Maç Sonucu",
                                    en_yaygin_sonuc,
                                    f"%{sonuc_oran:.0f}",
                                    help="Benzer maçlarda en sık görülen maç sonucu ve yüzdesi"
                                )
                            
                            with col5:
                                # İlk yarı sonucu analizi
                                iy_sonuclari = []
                                for _, row in result_df.iloc[1:].iterrows():
                                    if row['İY'] != '-':
                                        iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                        if iy_ev > iy_dep:
                                            iy_sonuclari.append("Ev Sahibi")
                                        elif iy_ev < iy_dep:
                                            iy_sonuclari.append("Deplasman")
                                        else:
                                            iy_sonuclari.append("Beraberlik")
                                
                                iy_dagilimi = pd.Series(iy_sonuclari).value_counts()
                                en_yaygin_iy_sonuc = iy_dagilimi.index[0] if not iy_dagilimi.empty else "-"
                                iy_sonuc_oran = (iy_dagilimi.iloc[0] / len(iy_sonuclari) * 100) if not iy_dagilimi.empty else 0
                                
                                st.metric(
                                    "En Yaygın İY Sonucu",
                                    en_yaygin_iy_sonuc,
                                    f"%{iy_sonuc_oran:.0f}",
                                    help="Benzer maçlarda en sık görülen ilk yarı sonucu ve yüzdesi"
                                )
                            
                            with col6:
                                # İkinci yarı sonucu analizi
                                iy2_sonuclari = []
                                for _, row in result_df.iloc[1:].iterrows():
                                    if row['İY'] != '-' and row['Skor'] != '-':
                                        # İlk yarı skorları
                                        iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                        # Maç sonu skorları
                                        ms_ev, ms_dep = map(int, row['Skor'].split('-'))
                                        # İkinci yarı skorları
                                        iy2_ev = ms_ev - iy_ev
                                        iy2_dep = ms_dep - iy_dep
                                        
                                        if iy2_ev > iy2_dep:
                                            iy2_sonuclari.append("Ev Sahibi")
                                        elif iy2_ev < iy2_dep:
                                            iy2_sonuclari.append("Deplasman")
                                        else:
                                            iy2_sonuclari.append("Beraberlik")
                                
                                iy2_dagilimi = pd.Series(iy2_sonuclari).value_counts()
                                en_yaygin_iy2_sonuc = iy2_dagilimi.index[0] if not iy2_dagilimi.empty else "-"
                                iy2_sonuc_oran = (iy2_dagilimi.iloc[0] / len(iy2_sonuclari) * 100) if not iy2_dagilimi.empty else 0
                                
                                st.metric(
                                    "En Yaygın 2Y Sonucu",
                                    en_yaygin_iy2_sonuc,
                                    f"%{iy2_sonuc_oran:.0f}",
                                    help="Benzer maçlarda en sık görülen ikinci yarı sonucu ve yüzdesi"
                                )

                            # Grafikler için sekmeler
                            tab1, tab2, tab3, tab4 = st.tabs(["Detaylı İstatistikler", "Gol Analizi", "İY/MS Analizi", "Form Analizi"])
                            
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
                                    mac_bilgileri = []
                                    
                                    for _, row in result_df.iloc[1:].iterrows():
                                        if row['İY'] != '-' and row['Skor'] != '-':
                                            iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                            ms_ev, ms_dep = map(int, row['Skor'].split('-'))
                                            iy_toplam = iy_ev + iy_dep
                                            iy2_toplam = (ms_ev + ms_dep) - (iy_ev + iy_dep)
                                            
                                            mac_bilgileri.append({
                                                'Maç': f"{row['Ev Sahibi']}-{row['Deplasman']}",
                                                'İlk Yarı': iy_toplam,
                                                'İkinci Yarı': iy2_toplam
                                            })
                                    
                                    df_yari = pd.DataFrame(mac_bilgileri)
                                    
                                    fig = px.bar(df_yari, 
                                               x='Maç',
                                               y=['İlk Yarı', 'İkinci Yarı'],
                                               title="Yarı Bazında Gol Dağılımı",
                                               barmode='group',
                                               color_discrete_sequence=['#FF9999', '#66B2FF'])
                                    
                                    fig.update_layout(
                                        xaxis_tickangle=-45,
                                        legend_title=None,
                                        height=400,
                                        margin=dict(t=30, b=100)  # Alt kısımda takım isimleri için daha fazla boşluk
                                    )
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

                            with col1:
                                st.write("##### Maç Maç Gol Dağılımı")
                                for idx, row in result_df.iloc[1:].iterrows():
                                    if row['Skor'] != '-':
                                        ev_gol, dep_gol = map(int, row['Skor'].split('-'))
                                        toplam_gol = ev_gol + dep_gol
                                        st.write(f"• {row['Ev Sahibi']} {row['Skor']} {row['Deplasman']}: **{toplam_gol}** gol")

                            with col2:
                                st.write("##### Maç Maç İlk Yarı Golleri")
                                for idx, row in result_df.iloc[1:].iterrows():
                                    if row['İY'] != '-':
                                        iy_ev, iy_dep = map(int, row['İY'].split('-'))
                                        iy_toplam = iy_ev + iy_dep
                                        st.write(f"• {row['Ev Sahibi']} {row['İY']} {row['Deplasman']}: **{iy_toplam}** gol")

                            # Yapay Zeka Analiz Bölümü
                            with st.container():
                                st.write("### 🤖 Maç Analizi")
                                
                                # Seçilen maç bilgileri
                                secilen_mac = result_df.iloc[0]
                                ev_sahibi = secilen_mac['Ev Sahibi']
                                deplasman = secilen_mac['Deplasman']
                                
                                # Benzer maçların analizi
                                benzer_maclar = result_df.iloc[1:]
                                
                                # Gol eğilimleri
                                ust_25_oran = len([g for g in goller if g > 2]) / len(goller) * 100
                                iy_15_oran = len([g for g in iy_goller if g > 1]) / len(iy_goller) * 100 if iy_goller else 0
                                kg_oran = kg_var / toplam_mac * 100 if toplam_mac > 0 else 0
                                
                                # Sonuç eğilimi
                                sonuc_egilimi = sonuc_dagilimi.index[0] if not sonuc_dagilimi.empty else "Belirsiz"
                                
                                # Analiz metni oluştur
                                analiz = f"""
                                    {ev_sahibi} - {deplasman} maçı için bulunan en benzer 5 maç analiz edildiğinde şu sonuçlara ulaşılmıştır:

                                    Benzer maçların {en_yaygin_aralik.lower()} ile sonuçlanma eğilimi %{en_yaygin_oran:.0f} oranında öne çıkmaktadır. İlk yarılarda ise en sık {en_yaygin_iy.lower()} görülme oranı %{en_yaygin_iy_oran:.0f}'dir. 

                                    Benzer maçlarda {sonuc_egilimi.lower()} sonucu ağır basmaktadır. Maçların %{ust_25_oran:.0f}'i 2.5 üstü golle sonuçlanırken, %{kg_oran:.0f}'inde karşılıklı gol görülmüştür. İlk yarılarda 1.5 üstü gol görülme oranı %{iy_15_oran:.0f}'dir.

                                    En yüksek benzerlik oranına sahip maç %{result_df.iloc[1]['Similarity']*100:.0f} benzerlik ile {result_df.iloc[1]['Ev Sahibi']} - {result_df.iloc[1]['Deplasman']} karşılaşmasıdır ve bu maç {result_df.iloc[1]['Skor']} sonuçlanmıştır.
                                """
                                st.write(analiz)

                            with tab4:
                                st.write("##### Form Analizi")
                                
                                # Ev sahibinin son 5 maçı
                                st.write(f"**{ev_sahibi} Son 5 Maçı**")
                                ev_son5 = df[
                                    ((df['Ev Sahibi'] == ev_sahibi) | (df['Deplasman'] == ev_sahibi))
                                ].copy()
                                
                                # Tarih karşılaştırması için tarihleri datetime formatına çevir
                                ev_son5['Tarih'] = pd.to_datetime(ev_son5['Tarih'], format='%d.%m.%Y')
                                secilen_tarih = pd.to_datetime(result_df.iloc[0]['Tarih'], format='%d.%m.%Y')
                                
                                # Seçilen maçtan önceki maçları filtrele
                                ev_son5 = ev_son5[ev_son5['Tarih'] < secilen_tarih]
                                ev_son5 = ev_son5.sort_values('Tarih', ascending=False).head(5)
                                
                                # Tarihi tekrar string formatına çevir
                                ev_son5['Tarih'] = ev_son5['Tarih'].dt.strftime('%d.%m.%Y')
                                
                                # Sonuç hesaplama
                                def get_team_result(row, team):
                                    if row['Skor'] == '-':
                                        return '⚪ Oynanmadı'
                                    ev_gol, dep_gol = map(int, row['Skor'].split('-'))
                                    if row['Ev Sahibi'] == team:
                                        if ev_gol > dep_gol: return '🟢 Kazandı'
                                        elif ev_gol < dep_gol: return '🔴 Kaybetti'
                                    else:  # Deplasman takımı
                                        if dep_gol > ev_gol: return '🟢 Kazandı'
                                        elif dep_gol < ev_gol: return '🔴 Kaybetti'
                                    return '⚪ Berabere'
                                
                                ev_son5['Sonuç'] = ev_son5.apply(lambda x: get_team_result(x, ev_sahibi), axis=1)
                                
                                if not ev_son5.empty:
                                    st.dataframe(ev_son5[['Tarih', 'Ev Sahibi', 'Skor', 'Deplasman', 'Sonuç']])
                                else:
                                    st.info(f"{ev_sahibi} için geçmiş maç verisi bulunamadı.")

                                # Deplasman takımının son 5 maçı
                                st.write(f"**{deplasman} Son 5 Maçı**")
                                dep_son5 = df[
                                    ((df['Ev Sahibi'] == deplasman) | (df['Deplasman'] == deplasman))
                                ].copy()
                                
                                dep_son5['Tarih'] = pd.to_datetime(dep_son5['Tarih'], format='%d.%m.%Y')
                                dep_son5 = dep_son5[dep_son5['Tarih'] < secilen_tarih]
                                dep_son5 = dep_son5.sort_values('Tarih', ascending=False).head(5)
                                dep_son5['Tarih'] = dep_son5['Tarih'].dt.strftime('%d.%m.%Y')
                                
                                dep_son5['Sonuç'] = dep_son5.apply(lambda x: get_team_result(x, deplasman), axis=1)
                                
                                if not dep_son5.empty:
                                    st.dataframe(dep_son5[['Tarih', 'Ev Sahibi', 'Skor', 'Deplasman', 'Sonuç']])
                                else:
                                    st.info(f"{deplasman} için geçmiş maç verisi bulunamadı.")

                                # Ev sahibinin iç saha son 5 maçı
                                st.write(f"**{ev_sahibi} Son 5 İç Saha Maçı**")
                                ev_ic_saha = df[df['Ev Sahibi'] == ev_sahibi].copy()
                                
                                ev_ic_saha['Tarih'] = pd.to_datetime(ev_ic_saha['Tarih'], format='%d.%m.%Y')
                                ev_ic_saha = ev_ic_saha[ev_ic_saha['Tarih'] < secilen_tarih]
                                ev_ic_saha = ev_ic_saha.sort_values('Tarih', ascending=False).head(5)
                                ev_ic_saha['Tarih'] = ev_ic_saha['Tarih'].dt.strftime('%d.%m.%Y')
                                
                                def get_home_result(skor):
                                    if skor == '-': return '⚪ Oynanmadı'
                                    ev_gol, dep_gol = map(int, skor.split('-'))
                                    if ev_gol > dep_gol: return '🟢 Kazandı'
                                    elif ev_gol < dep_gol: return '🔴 Kaybetti'
                                    return '⚪ Berabere'
                                
                                ev_ic_saha['Sonuç'] = ev_ic_saha['Skor'].apply(get_home_result)
                                
                                if not ev_ic_saha.empty:
                                    st.dataframe(ev_ic_saha[['Tarih', 'Ev Sahibi', 'Skor', 'Deplasman', 'Sonuç']])
                                else:
                                    st.info(f"{ev_sahibi} için iç saha maç verisi bulunamadı.")

                                # Deplasman takımının deplasman son 5 maçı
                                st.write(f"**{deplasman} Son 5 Deplasman Maçı**")
                                dep_deplasman = df[df['Deplasman'] == deplasman].copy()
                                
                                dep_deplasman['Tarih'] = pd.to_datetime(dep_deplasman['Tarih'], format='%d.%m.%Y')
                                dep_deplasman = dep_deplasman[dep_deplasman['Tarih'] < secilen_tarih]
                                dep_deplasman = dep_deplasman.sort_values('Tarih', ascending=False).head(5)
                                dep_deplasman['Tarih'] = dep_deplasman['Tarih'].dt.strftime('%d.%m.%Y')
                                
                                def get_away_result(skor):
                                    if skor == '-': return '⚪ Oynanmadı'
                                    ev_gol, dep_gol = map(int, skor.split('-'))
                                    if dep_gol > ev_gol: return '🟢 Kazandı'
                                    elif dep_gol < ev_gol: return '🔴 Kaybetti'
                                    return '⚪ Berabere'
                                
                                dep_deplasman['Sonuç'] = dep_deplasman['Skor'].apply(get_away_result)
                                
                                if not dep_deplasman.empty:
                                    st.dataframe(dep_deplasman[['Tarih', 'Ev Sahibi', 'Skor', 'Deplasman', 'Sonuç']])
                                else:
                                    st.info(f"{deplasman} için deplasman maç verisi bulunamadı.")

if __name__ == "__main__":
    main()
