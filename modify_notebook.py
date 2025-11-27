import json
import os

notebook_path = r"c:\Users\asjer\OneDrive\Documentos\4GeeksAcademy\machine-learning-nlp\src\explore.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Define new cells
cell_viz_dist_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Visualización de la distribución de clases\n", "- Es importante ver qué tan desbalanceado está el dataset."]
}

cell_viz_dist_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "plt.figure(figsize=(6, 4))\n",
        "sns.countplot(x='is_spam', data=total_data)\n",
        "plt.title('Distribución de Clases: Spam vs No-Spam')\n",
        "plt.xlabel('Es Spam? (0=No, 1=Si)')\n",
        "plt.ylabel('Cantidad')\n",
        "plt.show()"
    ] 
}

cell_viz_wc_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Nube de palabras (WordCloud)\n", "- Visualizamos los términos más frecuentes en URLs de Spam vs Legítimas."]
}

cell_viz_wc_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from wordcloud import WordCloud\n",
        "\n",
        "# Separar URLs de spam y no spam\n",
        "spam_text = \" \".join(total_data[total_data['is_spam'] == 1]['url_cleaned'])\n",
        "non_spam_text = \" \".join(total_data[total_data['is_spam'] == 0]['url_cleaned'])\n",
        "\n",
        "# Generar nubes de palabras\n",
        "wordcloud_spam = WordCloud(width=800, height=400, background_color='black').generate(spam_text)\n",
        "wordcloud_non_spam = WordCloud(width=800, height=400, background_color='white').generate(non_spam_text)\n",
        "\n",
        "# Graficar\n",
        "plt.figure(figsize=(15, 7))\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.imshow(wordcloud_spam, interpolation='bilinear')\n",
        "plt.axis('off')\n",
        "plt.title('Nube de Palabras - SPAM')\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.imshow(wordcloud_non_spam, interpolation='bilinear')\n",
        "plt.axis('off')\n",
        "plt.title('Nube de Palabras - NO SPAM')\n",
        "\n",
        "plt.show()"
    ]
}

cell_viz_cm_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Visualización del rendimiento (Matriz de Confusión)\n", "- Evaluamos dónde se equivoca más el modelo."]
}

cell_viz_cm_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "\n",
        "# Predecir con el mejor modelo\n",
        "y_pred_best = best_model.predict(X_test)\n",
        "\n",
        "# Matriz de confusión\n",
        "cm = confusion_matrix(y_test, y_pred_best)\n",
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[\"No Spam\", \"Spam\"])\n",
        "disp.plot(cmap=plt.cm.Blues)\n",
        "plt.title(\"Matriz de Confusión (Mejor Modelo)\")\n",
        "plt.show()"
    ]
}

# Insert cells
new_cells = []
for cell in cells:
    new_cells.append(cell)
    source = "".join(cell.get("source", []))
    
    # Insert Class Balance after value_counts
    if 'total_data["is_spam"].value_counts()' in source:
        new_cells.append(cell_viz_dist_md)
        new_cells.append(cell_viz_dist_code)
        
    # Insert WordCloud after url_cleaned creation
    if 'total_data["url_cleaned"] = total_data["url"].apply(preprocess_url)' in source:
        new_cells.append(cell_viz_wc_md)
        new_cells.append(cell_viz_wc_code)
        
    # Insert Confusion Matrix after grid_search.fit
    if 'grid_search.fit(X_train, y_train)' in source:
        new_cells.append(cell_viz_cm_md)
        new_cells.append(cell_viz_cm_code)

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
