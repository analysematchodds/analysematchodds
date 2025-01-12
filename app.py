import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(csv_path, selected_columns):
    # CSV'yi oku
    df = pd.read_csv(csv_path)
    
    # Tüm sayısal sütunları belirle
    numeric_columns = ['MS1', 'MS0', 'MS2', 'AU2.5 Alt', 'AU2.5 Üst', 
                      'KG Var', 'KG Yok', 'IY0.5 Alt', 'IY0.5 Üst',
                      'AU1.5 Alt', 'AU1.5 Üst', 'IY1', 'IY0', 'IY2',
                      '2Y1', '2Y0', '2Y2', 'Tek', 'Çift']
    
    # Tüm sayısal sütunları float'a çevir ve yuvarla
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2).fillna(0)
    
    # Seçilen sütunlar için eksik ve '-' değerlerini filtrele
    for col in selected_columns:
        df = df[df[col] != '-']  # '-' değerlerini kaldır
        
    df.fillna(0, inplace=True)
    
    return df

def calculate_similarity(df, numeric_columns):
    # Verileri normalize et
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[numeric_columns])
    
    # Benzerlik matrisi oluştur
    n_matches = len(df)
    similarity_matrix = np.zeros((n_matches, n_matches))
    
    # Her maç çifti için benzerlik hesapla
    for i in range(n_matches):
        for j in range(i+1, n_matches):
            # Oranlar arasındaki farkları hesapla
            diff = np.abs(normalized_data[i] - normalized_data[j])
            
            # Ağırlıklı benzerlik skoru hesapla
            if all(col in numeric_columns for col in ['MS1', 'MS0', 'MS2']):
                ms_idx = [numeric_columns.index(col) for col in ['MS1', 'MS0', 'MS2']]
                
                # Diğer oranların sayısını kontrol et
                other_count = len(numeric_columns) - 3
                if other_count > 0:
                    ms_weight = 0.6
                    other_weight = 0.4 / other_count
                else:
                    # Sadece MS oranları varsa eşit ağırlık ver
                    ms_weight = 1/3
                    other_weight = 1/3
                
                weights = np.array([
                    ms_weight if idx in ms_idx else other_weight 
                    for idx in range(len(numeric_columns))
                ])
            else:
                # Tüm oranlara eşit ağırlık ver
                weights = np.ones(len(numeric_columns)) / len(numeric_columns)
            
            # Ağırlıklı ortalama fark
            weighted_diff = np.average(diff, weights=weights)
            
            # Benzerlik skoru (fark azaldıkça benzerlik artar)
            similarity = 1 / (1 + weighted_diff)
            
            similarity_matrix[i,j] = similarity
            similarity_matrix[j,i] = similarity
    
    return similarity_matrix

def find_similar_matches(df, similarity_matrix, selected_idx, n_matches=5):
    # Seçilen maça benzerlik skorlarını al
    similarities = similarity_matrix[selected_idx]
    
    # En benzer n maçın indexlerini bul (kendisi hariç)
    similar_indices = np.argsort(similarities)[::-1][1:n_matches+1]
    
    # Seçilen maç ve benzer maçları DataFrame'e ekle
    matches = [df.iloc[selected_idx]]  # Seçilen maç ilk sırada
    similarities_list = [1.0]  # Kendisiyle benzerliği 1.0
    
    for idx in similar_indices:
        matches.append(df.iloc[idx])
        similarities_list.append(similarities[idx])
    
    result_df = pd.concat(matches, axis=1).T.reset_index(drop=True)
    result_df['Similarity'] = similarities_list
    
    # Tüm sayısal sütunları belirle
    numeric_columns = [
        'MS1', 'MS0', 'MS2', 'AU2.5 Alt', 'AU2.5 Üst', 
        'KG Var', 'KG Yok', 'IY0.5 Alt', 'IY0.5 Üst',
        'AU1.5 Alt', 'AU1.5 Üst', 'IY1', 'IY0', 'IY2',
        '2Y1', '2Y0', '2Y2', 'Tek', 'Çift',
        'IY Çifte Şans 1-X', 'IY Çifte Şans 1-2', 'IY Çifte Şans X-2',
        'IY/MS 1/1', 'IY/MS 1/0', 'IY/MS 1/2', 'IY/MS 0/1', 
        'IY/MS 0/0', 'IY/MS 0/2', 'IY/MS 2/1', 'IY/MS 2/0', 'IY/MS 2/2',
        'Çifte Şans 1-X', 'Çifte Şans 1-2', 'Çifte Şans X-2'
    ]
    
    # Tüm sayısal sütunları yuvarla
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').round(2)
    
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
            'Tek/Çift': ['Tek', 'Çift']
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
        
    # Arama kutusu
    search_term = st.text_input("Takım adı ile ara:", "")
    
    # CSV dosyasını seçilen kriterlere göre oku
    df = load_and_prepare_data('https://raw.githubusercontent.com/analysematchodds/match_odds_csv/refs/heads/main/matchodds.csv', selected_criteria)
    
    df['Tarih'] = pd.to_datetime(df['Tarih']).dt.strftime('%d.%m.%Y')
    df['Saat'] = pd.to_datetime(df['Saat'], format='%H:%M:%S').dt.strftime('%H:%M')

    # Benzerlik matrisini hesapla
    similarity_matrix = calculate_similarity(df, selected_criteria)
    
    if search_term:
        # Takım adına göre filtrele
        mask = (
            df['Ev Sahibi'].str.contains(search_term, case=False, na=False) |
            df['Deplasman'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = df[mask]
        
        if len(filtered_df) == 0:
            st.warning("Eşleşen maç bulunamadı!")
        else:
            st.write(f"{len(filtered_df)} maç bulundu:")
            
            # Maçları göster ve seçim yaptır
            matches = []
            for idx, row in filtered_df.iterrows():
                odds_str = "/".join(f"{row[col]:.2f}" for col in selected_criteria)
                match_str = f"{row['Ev Sahibi']} vs {row['Deplasman']} ({row['Tarih']} - {row['Saat']}) - Seçili Oranlar: {odds_str}"
                matches.append((df.index.get_loc(idx), match_str))
            
            selected_match = st.selectbox(
                "Maç seçin:",
                options=matches,
                format_func=lambda x: x[1]
            )
            
            if st.button("Benzer Maçları Bul"):
                selected_idx = selected_match[0]
                result_df = find_similar_matches(df, similarity_matrix, selected_idx)
                
                # Format sözlüğünü tüm sayısal sütunlar için oluştur
                format_dict = {
                    'MS1': '{:.2f}', 'MS0': '{:.2f}', 'MS2': '{:.2f}',
                    'AU2.5 Alt': '{:.2f}', 'AU2.5 Üst': '{:.2f}',
                    'KG Var': '{:.2f}', 'KG Yok': '{:.2f}',
                    'IY0.5 Alt': '{:.2f}', 'IY0.5 Üst': '{:.2f}',
                    'AU1.5 Alt': '{:.2f}', 'AU1.5 Üst': '{:.2f}',
                    'IY1': '{:.2f}', 'IY0': '{:.2f}', 'IY2': '{:.2f}',
                    '2Y1': '{:.2f}', '2Y0': '{:.2f}', '2Y2': '{:.2f}',
                    'Tek': '{:.2f}', 'Çift': '{:.2f}',
                    'IY Çifte Şans 1-X': '{:.2f}', 'IY Çifte Şans 1-2': '{:.2f}', 'IY Çifte Şans X-2': '{:.2f}',
                    'IY/MS 1/1': '{:.2f}', 'IY/MS 1/0': '{:.2f}', 'IY/MS 1/2': '{:.2f}',
                    'IY/MS 0/1': '{:.2f}', 'IY/MS 0/0': '{:.2f}', 'IY/MS 0/2': '{:.2f}',
                    'IY/MS 2/1': '{:.2f}', 'IY/MS 2/0': '{:.2f}', 'IY/MS 2/2': '{:.2f}',
                    'Çifte Şans 1-X': '{:.2f}', 'Çifte Şans 1-2': '{:.2f}', 'Çifte Şans X-2': '{:.2f}',
                    'Similarity': '{:.2%}'
                }
                
                # Seçilen maçı göster
                st.write("### Seçilen Maç")
                st.dataframe(
                    result_df.iloc[[0]].style.format(format_dict)
                )
                
                # Tüm detaylarla benzer maçları göster
                st.write("### En Benzer 5 Maç (Tüm Detaylar)")
                st.dataframe(
                    result_df.iloc[1:].style.format(format_dict)
                )
                
                # Sadece seçili kriterleri içeren benzer maçları göster
                st.write("### En Benzer 5 Maç (Seçili Kriterler)")
                
                # Temel bilgi sütunları + seçili kriterler + benzerlik skoru
                selected_columns = ['Tarih', 'Saat', 'Lig', 'MBS', 'Ev Sahibi', 'Skor', 'Deplasman', 'İY'] + selected_criteria + ['Similarity']
                
                # Format sözlüğünü seçili kriterler için güncelle
                selected_format_dict = {col: '{:.2f}' for col in selected_criteria}
                selected_format_dict['Similarity'] = '{:.2%}'
                
                st.dataframe(
                    result_df.iloc[1:][selected_columns].style.format(selected_format_dict)
                )
                
                # CSV indirme butonu
                csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="Sonuçları CSV olarak indir",
                    data=csv,
                    file_name="similar_matches.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()