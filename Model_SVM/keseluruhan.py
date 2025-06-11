import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.graph_objs as go
import plotly.figure_factory as ff
import joblib
import matplotlib.pyplot as plt

# Load dataset dari CSV
df = pd.read_csv('dataset_kategori_3kelas.csv')  # Pastikan file ada di folder kerja

print("Data sample:")
print(df.head())

print("\nDistribusi kelas:")
print(df['Kelas'].value_counts())

# Plot distribusi data (BPM vs Kemiringan) berdasarkan kelas
colors = {1:'green', 2:'orange', 3:'red'}
plt.figure(figsize=(10,6))
for kelas in df['Kelas'].unique():
    subset = df[df['Kelas'] == kelas]
    plt.scatter(subset['BPM'], subset['Kemiringan'], 
                c=colors[kelas], label=f'Kelas {kelas}', alpha=0.6, edgecolors='k')
plt.xlabel('BPM')
plt.ylabel('Kemiringan')
plt.title('Distribusi Data per Kelas')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data untuk modeling
X = df[['BPM', 'Kemiringan']]
y = df['Kelas']

# Split data 90% train, 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y)

print("\nUkuran dataset:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Train SVM dengan kernel RBF
svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X_train, y_train)

# Save model
joblib.dump(svm_model, 'svm_model.pkl')

# Prediksi test set
y_pred = svm_model.predict(X_test)

# Hitung akurasi
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi: {acc:.4f}")

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
labels = ['Normal', 'Mengantuk Sedang', 'Mengantuk Berat']

fig1 = ff.create_annotated_heatmap(
    z=conf_mat,
    x=labels,
    y=labels,
    colorscale='Blues',
    showscale=True
)
fig1.update_layout(title='Confusion Matrix',
                   xaxis_title='Predicted Label',
                   yaxis_title='True Label')

# Plot decision boundary
x_min, x_max = X['BPM'].min() - 5, X['BPM'].max() + 5
y_min, y_max = X['Kemiringan'].min() - 5, X['Kemiringan'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

colors_plotly = ['#1f77b4', '#ff7f0e', '#2ca02c']
decision_surface = go.Contour(
    x=np.arange(x_min, x_max, 1),
    y=np.arange(y_min, y_max, 1),
    z=Z,
    colorscale='Viridis',
    opacity=0.3,
    showscale=False,
    contours=dict(showlines=False)
)

scatter = go.Scatter(
    x=X['BPM'],
    y=X['Kemiringan'],
    mode='markers',
    marker=dict(color=[colors_plotly[int(c)-1] for c in y],
                size=7,
                line=dict(width=1, color='black')),
    text=[f'Kelas: {labels[int(c)-1]}' for c in y],
    name='Data Points'
)

fig2 = go.Figure(data=[decision_surface, scatter])
fig2.update_layout(
    title='Decision Boundary Interaktif',
    xaxis_title='BPM',
    yaxis_title='Kemiringan',
    legend_title='Kelas'
)

# Tampilkan grafik interaktif
fig1.show()
fig2.show()
