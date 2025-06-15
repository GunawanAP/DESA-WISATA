from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os
import random

app = Flask(__name__)

# Load CSV relatif ke path file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "desa_wisata_cleaned.csv")
df = pd.read_csv(csv_path)

# Simulasi koordinat jika tidak ada kolom latitude/longitude
# Dalam implementasi nyata, koordinat harus ada di CSV
def generate_coordinates(kabupaten):
    """Generate random coordinates for demo purposes"""
    # Koordinat approximate untuk beberapa kabupaten di Indonesia
    kabupaten_coords = {
        'default': {'lat': -6.2, 'lng': 106.8},  # Jakarta area
        'yogyakarta': {'lat': -7.8, 'lng': 110.4},
        'bandung': {'lat': -6.9, 'lng': 107.6},
        'malang': {'lat': -7.9, 'lng': 112.6},
        'banyuwangi': {'lat': -8.2, 'lng': 114.4},
        'lombok': {'lat': -8.6, 'lng': 116.3},
        'bali': {'lat': -8.3, 'lng': 115.1},
    }
    
    # Cari koordinat base berdasarkan nama kabupaten
    base_coord = kabupaten_coords['default']
    for key, coord in kabupaten_coords.items():
        if key.lower() in str(kabupaten).lower():
            base_coord = coord
            break
    
    # Tambah random offset kecil untuk variasi
    lat_offset = random.uniform(-0.5, 0.5)
    lng_offset = random.uniform(-0.5, 0.5)
    
    return {
        'lat': base_coord['lat'] + lat_offset,
        'lng': base_coord['lng'] + lng_offset
    }

# Tambahkan koordinat jika belum ada
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    coordinates = df['NAMA KABUPATEN'].apply(generate_coordinates)
    df['latitude'] = [coord['lat'] for coord in coordinates]
    df['longitude'] = [coord['lng'] for coord in coordinates]

# Inisialisasi stopword remover
stop_factory = StopWordRemoverFactory()
stop_remover = stop_factory.create_stop_word_remover()

# Membersihkan teks dari stopword
def preprocess(text):
    if pd.isna(text):
        return ""
    return stop_remover.remove(text.lower())

# Terapkan preprocessing ke kolom deskripsi
df['cleaned_profil'] = df['profil_desa'].apply(preprocess)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['cleaned_profil'])

# Fungsi untuk mengkategorikan desa berdasarkan profil
def categorize_desa(profil_text):
    if pd.isna(profil_text):
        return "Lainnya"
    
    text_lower = profil_text.lower()
    categories = []
    
    # Kategori wisata alam
    alam_keywords = ['alam', 'pantai', 'gunung', 'hutan', 'air terjun', 'danau', 'sungai', 'pemandangan', 'hiking', 'trekking']
    if any(keyword in text_lower for keyword in alam_keywords):
        categories.append('Alam')
    
    # Kategori wisata budaya
    budaya_keywords = ['budaya', 'adat', 'tradisi', 'sejarah', 'heritage', 'museum', 'candi', 'upacara', 'kesenian', 'tari']
    if any(keyword in text_lower for keyword in budaya_keywords):
        categories.append('Budaya')
    
    # Kategori wisata kuliner
    kuliner_keywords = ['kuliner', 'makanan', 'masakan', 'khas', 'olahan', 'cemilan', 'jajanan', 'restoran', 'warung']
    if any(keyword in text_lower for keyword in kuliner_keywords):
        categories.append('Kuliner')
    
    # Kategori wisata religi
    religi_keywords = ['religi', 'masjid', 'gereja', 'pura', 'vihara', 'makam', 'ziarah', 'spiritual']
    if any(keyword in text_lower for keyword in religi_keywords):
        categories.append('Religi')
    
    # Kategori wisata edukasi
    edukasi_keywords = ['edukasi', 'belajar', 'pendidikan', 'pelatihan', 'workshop', 'keterampilan', 'kerajinan']
    if any(keyword in text_lower for keyword in edukasi_keywords):
        categories.append('Edukasi')
    
    return ', '.join(categories) if categories else 'Lainnya'

# Tambahkan kolom kategori
df['kategori_wisata'] = df['profil_desa'].apply(categorize_desa)

@app.route('/')
def index():
    # Ambil daftar kabupaten dan kategori untuk dropdown
    kabupaten_list = sorted(df['NAMA KABUPATEN'].dropna().unique())
    kategori_list = sorted(df['kategori_wisata'].dropna().unique())
    
    return render_template('index.html', 
                         kabupaten_list=kabupaten_list, 
                         kategori_list=kategori_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    preferences = request.form.get('preferences', '')
    query = request.form.get('preferences', '')
    selected_kategori = request.form.get('kategori', 'semua')
    selected_kabupaten = request.form.get('kabupaten', 'semua')

    
    # Filter dataframe berdasarkan pilihan user
    filtered_df = df.copy()
    
    if selected_kabupaten and selected_kabupaten != 'semua':
        filtered_df = filtered_df[filtered_df['NAMA KABUPATEN'] == selected_kabupaten]
    
    if selected_kategori and selected_kategori != 'semua':
        filtered_df = filtered_df[filtered_df['kategori_wisata'].str.contains(selected_kategori, na=False)]
    
    if filtered_df.empty:
        return render_template('results.html', 
                             preferences=query, 
                             results=[], 
                             message="Tidak ada desa wisata yang sesuai dengan filter yang dipilih.")
    
    # Jika ada query pencarian, gunakan similarity
    if query.strip():
        preferences = [pref.strip() for pref in query.split(',') if pref.strip()]
        combined_query = ' '.join(preferences)
        clean_query = preprocess(combined_query)
        
        # Buat TF-IDF matrix baru untuk data yang sudah difilter
        filtered_tfidf = vectorizer.transform(filtered_df['cleaned_profil'])
        query_vec = vectorizer.transform([clean_query])
        similarity = cosine_similarity(query_vec, filtered_tfidf).flatten()
        
        # Ambil top 10 hasil
        top_indices = similarity.argsort()[-10:][::-1]
        # Filter hanya yang memiliki similarity > 0
        relevant_indices = [i for i in top_indices if similarity[i] > 0][:5]
        
        if not relevant_indices:
            # Jika tidak ada yang relevan, ambil 5 teratas dari filter
            results = filtered_df.head(5)[['NAMA DESA', 'NAMA KABUPATEN', 'profil_desa', 'kategori_wisata', 'latitude', 'longitude']].to_dict(orient='records')
        else:
            results = filtered_df.iloc[relevant_indices][['NAMA DESA', 'NAMA KABUPATEN', 'profil_desa', 'kategori_wisata', 'latitude', 'longitude']].to_dict(orient='records')
    else:
        # Jika tidak ada query, tampilkan hasil filter saja
        results = filtered_df.head(10)[['NAMA DESA', 'NAMA KABUPATEN', 'profil_desa', 'kategori_wisata', 'latitude', 'longitude']].to_dict(orient='records')
    
    return render_template(
        'results.html',
        results=results,
        preferences=preferences,
        selected_kategori=selected_kategori,
        selected_kabupaten=selected_kabupaten)

@app.route('/api/kabupaten/<kategori>')
def get_kabupaten_by_kategori(kategori):
    """API untuk mendapat daftar kabupaten berdasarkan kategori"""
    if kategori == 'semua':
        kabupaten_list = sorted(df['NAMA KABUPATEN'].dropna().unique())
    else:
        filtered_df = df[df['kategori_wisata'].str.contains(kategori, na=False)]
        kabupaten_list = sorted(filtered_df['NAMA KABUPATEN'].dropna().unique())
    
    return jsonify(kabupaten_list)

@app.route('/map')
def show_map():
    """Halaman peta dengan semua desa wisata"""
    # Ambil parameter filter dari URL jika ada
    kategori = request.args.get('kategori', 'semua')
    kabupaten = request.args.get('kabupaten', 'semua')
    
    # Filter data sesuai parameter
    filtered_df = df.copy()
    
    if kabupaten != 'semua':
        filtered_df = filtered_df[filtered_df['NAMA KABUPATEN'] == kabupaten]
    
    if kategori != 'semua':
        filtered_df = filtered_df[filtered_df['kategori_wisata'].str.contains(kategori, na=False)]
    
    # Ambil data untuk peta (maksimal 100 untuk performa)
    map_data = filtered_df.head(100)[['NAMA DESA', 'NAMA KABUPATEN', 'profil_desa', 'kategori_wisata', 'latitude', 'longitude']].to_dict(orient='records')
    
    return render_template('map.html', results=map_data)

@app.route('/map/search', methods=['POST'])
def map_search():
    """Peta berdasarkan hasil pencarian"""
    query = request.form.get('preferences', '')
    selected_kabupaten = request.form.get('kabupaten', 'semua')
    selected_kategori = request.form.get('kategori', 'semua')
    
    # Filter dataframe berdasarkan pilihan user
    filtered_df = df.copy()
    
    if selected_kabupaten != 'semua':
        filtered_df = filtered_df[filtered_df['NAMA KABUPATEN'] == selected_kabupaten]
    
    if selected_kategori != 'semua':
        filtered_df = filtered_df[filtered_df['kategori_wisata'].str.contains(selected_kategori, na=False)]
    
    # Jika ada query pencarian, gunakan similarity
    if query.strip():
        preferences = [pref.strip() for pref in query.split(',') if pref.strip()]
        combined_query = ' '.join(preferences)
        clean_query = preprocess(combined_query)
        
        # Buat TF-IDF matrix baru untuk data yang sudah difilter
        filtered_tfidf = vectorizer.transform(filtered_df['cleaned_profil'])
        query_vec = vectorizer.transform([clean_query])
        similarity = cosine_similarity(query_vec, filtered_tfidf).flatten()
        
        # Ambil top 20 hasil untuk peta
        top_indices = similarity.argsort()[-20:][::-1]
        relevant_indices = [i for i in top_indices if similarity[i] > 0]
        
        if not relevant_indices:
            map_data = filtered_df.head(20)[['NAMA DESA', 'NAMA KABUPATEN', 'profil_desa', 'kategori_wisata', 'latitude', 'longitude']].to_dict(orient='records')
        else:
            map_data = filtered_df.iloc[relevant_indices][['NAMA DESA', 'NAMA KABUPATEN', 'profil_desa', 'kategori_wisata', 'latitude', 'longitude']].to_dict(orient='records')
    else:
        map_data = filtered_df.head(20)[['NAMA DESA', 'NAMA KABUPATEN', 'profil_desa', 'kategori_wisata', 'latitude', 'longitude']].to_dict(orient='records')
    
    return render_template('map.html', results=map_data)

if __name__ == '__main__':
    app.run(debug=True)