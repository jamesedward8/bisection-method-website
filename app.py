from flask import Flask, render_template, request, jsonify, send_file
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64

app = Flask(__name__)


#Route ke UI (file HTML)
@app.route('/')
def index():
    return render_template('index.html')


#Route untuk menghitung metode biseksi
@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        function = request.form['function']
        x_l = float(request.form['x_l'])
        x_u = float(request.form['x_u'])
        kriteria = request.form['kriteria']

        n = request.form['n'] if 'n' in request.form and request.form['n'] else None
        n_iterasi = request.form['n_iterasi'] if 'n_iterasi' in request.form and request.form['n_iterasi'] else None
        eps = request.form['eps'] if 'eps' in request.form and request.form['eps'] else None

        if n is not None:
            n = int(n)
        if n_iterasi is not None:
            n_iterasi = int(n_iterasi)
        if eps is not None:
            eps = float(eps)

        f_x_l = evaluate_function(function, x_l)
        f_x_u = evaluate_function(function, x_u)
        
        if f_x_l * f_x_u >= 0 or x_l == x_u:
            return jsonify({"error": "Nilai f(x_l) dan f(x_u) harus berubah tanda / Akar tidak ditemukan."})

        result, jumlah_iterasi, data = bisection_method(function, x_l, x_u, kriteria, n, n_iterasi, eps)
        graph_img = graph_function(function, x_l, x_u, result)
        return jsonify({
            "hasil": result,
            "jumlah_iterasi": jumlah_iterasi,
            "data": data,
            "graph": graph_img,
            "kriteria": kriteria
        })
    except Exception as e:
        return jsonify({"error": str(e)})
    

#Route untuk export file excel
@app.route('/export', methods=['POST'])
def export():
    try:
        data = request.json['data']
        
        column_order = [
            "iterasi",
            "x_l",
            "x_u",
            "f_x_l",
            "f_x_u",
            "x_r",
            "f_x_r",
            "abs f_x_r",  
            "eps_a"
        ]

        df = pd.DataFrame(data)[column_order]

        output = BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Bisection Results')

        output.seek(0) 

        return send_file(output, 
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                         as_attachment=True, 
                         download_name='bisection_results.xlsx')
    except Exception as e:
        return jsonify({"error": str(e)})
    

#Fungsi untuk mendapatkan nilai f(x) pada titik x 
def evaluate_function(function, x):
    x_symbol = sp.Symbol('x')
    function = function.replace('ln', 'log')
    function = sp.sympify(function)
    result = function.subs(x_symbol, x)
    return float(result) 


#Fungsi untuk mendapatkan grafik fungsi f(x) + akar persamaan
def graph_function(function, x_l, x_u, x_r):
    x_symbol = sp.Symbol('x')
    function = sp.sympify(function)
    
    f = sp.lambdify(x_symbol, function, 'numpy')
    
    x_vals = np.linspace(x_l - 1, x_u + 1, 400)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f'f(x) = {function}', color='purple', linewidth=1, linestyle='--')

    plt.scatter([x_l, x_u, x_r], [0, 0, 0], color=['red', 'green', 'blue'], s=100, zorder=5)
    
    offset = 2  
    plt.text(x_l, offset, f'x_l = {x_l}', color='red', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
    plt.text(x_u, offset, f'x_u = {x_u}', color='green', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
    plt.text(x_r, offset - 6, f'x_r = {x_r}', color='blue', verticalalignment='bottom', horizontalalignment='center', fontsize=10)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.title(f'Fungsi f(x) = {function} = 0', fontsize=12, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)

    plt.ylim(min(y_vals) - 1, max(y_vals) + 1)

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return img_base64


#Fungsi untuk perhitungan biseksi
def bisection_method(function, x_l, x_u, kriteria, n, n_iterasi, eps):
    eps_s = 0.5 * 10**(2-n)
    iter_count = 1
    x_r = 0 
    data = [] 
    
    while True:
        x_r_old = x_r 

        x_r = (x_l + x_u) / 2.0

        f_x_r = evaluate_function(function, x_r)
        f_x_l = evaluate_function(function, x_l)
        f_x_u = evaluate_function(function, x_u)
        
        if iter_count == 1:
            eps_a = "-"  
        else:
            eps_a = abs((x_r - x_r_old) / x_r * 100)
        
        data.append({
            "iterasi": iter_count,
            "x_l": x_l,
            "x_u": x_u,
            "f_x_l": f_x_l,
            "f_x_u": f_x_u,
            "x_r": x_r,
            "f_x_r": f_x_r,
            "abs f_x_r": abs(f_x_r),
            "eps_a": eps_a
        })
        
        if f_x_l * f_x_r < 0:
            x_u = x_r
        elif f_x_l * f_x_r > 0:
            x_l = x_r
        else:
            break

        if kriteria == 'n':
            if iter_count > 1 and eps_a < eps_s:
                break
        elif kriteria == 'n_iterasi':
            if iter_count >= n_iterasi:
                break
        elif kriteria == 'eps':
            if abs(f_x_r) < eps:
                break
        
        iter_count += 1

    return x_r, iter_count, data


if __name__ == '__main__':      
    app.run(debug=True)