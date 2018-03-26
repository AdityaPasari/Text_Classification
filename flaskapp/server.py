from flask import Flask,render_template,request, send_file, jsonify
import re,os,pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


app = Flask(__name__)
cur_dir = os.getcwd()
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','log_classifier.pkl'), 'rb'))
print(clf.best_params_)


@app.route('/')
def index():
    words = request.args.get("words")
    print(type(words))
    if(words is not None):
        words = words.split(',')
        prediction = clf.predict(words)
        probability =  clf.predict_proba(words)
        confidence = []
        for i in probability:
            confidence.append(max(i))
        print(confidence)
        return jsonify(prediction=prediction.tolist(),confidence = confidence)

    return render_template('home.html')


@app.route('/check_your_model')
def check_your_model():
    return render_template('check_your_model.html')


@app.route('/model_details')
def model_details():
    return render_template('model_details.html')


@app.route('/manual')
def manual():
    return render_template('manual.html')


@app.route('/predict_output',methods=['GET','POST'])
def predict_output():
    if request.method == 'POST':
        document = request.form['document']
        document=document.split('\n')
        results = clf.predict(document)
        probability = clf.predict_proba(document)
        confidence = []
        for i in probability:
            confidence.append(max(i))

        return jsonify(results = results.tolist(),confidence = confidence)

    return render_template('predict_output.html')


@app.route('/uploaded_predict_output',methods=['GET','POST'])
def uploaded_predict_output():
    if request.method == 'POST':
        destination  = os.path.join('uploaded_files')
        if not os.path.exists(destination):
            os.makedirs(destination)

        file_ = request.files.getlist("newfile")[0]
        filename = file_.filename
        dest = os.path.join('uploaded_files',filename)
        file_.save(dest)
        X = []
        with open(dest,'r') as data:
            for line in data:
                X.append(line.strip())
        results = clf.predict(X)
        probability = clf.predict_proba(X)
        confidence = []
        for i in probability:
            confidence.append(max(i))
        return jsonify(results = results.tolist(),confidence = confidence)

    return render_template('uploaded_predict_output.html')


@app.route('/predict_accuracy',methods=['GET','POST'])
def predict_accuracy():
    if request.method == 'POST':
        document = request.form['document']

        document=document.split('\n')
        X = []
        Y_actual = []
        for item in document:
            temp = item.split(',')
            if(len(temp) ==2):
                X.append(temp[1])
                Y_actual.append(temp[0])
            else:
                return render_template('data_error.html')
        results = clf.predict(X)
        probability = clf.predict_proba(X)
        confidence = []
        for i in probability:
            confidence.append(max(i))

        report = (classification_report(Y_actual,results)).split('\n')
        accuracy = np.mean(results == Y_actual)
        cnf_matrix = confusion_matrix(Y_actual, results)

        class_names = []
        for i in range(2,len(report)):
            class_names.append(''.join([i for i in report[i] if not i.isdigit()]).replace('.','').strip())

        new_cnf = []
        for i in range(len(cnf_matrix)):
            x = [class_names[i]]
            for item in cnf_matrix[i]:
                x.append(item)

            new_cnf.append(x)

        new__cnf = []
        for i in range(len(new_cnf)):
            new__cnf.append([])
            for j in range(len(new_cnf[i])):
                if(type(new_cnf[i][j]) is not str):
                    new__cnf[i].append(float(new_cnf[i][j]))
                else:
                    new__cnf[i].append(new_cnf[i][j])

        if(request.form.get('view_results')):
            return jsonify(accuracy=accuracy,total = len(X),report = report,confusion_matrix = new__cnf,results = results.tolist(),confidence = confidence)
        else:
            return jsonify(accuracy=accuracy,total = len(X),report = report,confusion_matrix = new__cnf,confidence = confidence)

    return render_template('predict_accuracy.html')


@app.route('/uploaded_predict_accuracy',methods=['GET','POST'])
def uploaded_predict_accuracy():

    if request.method == 'POST':
        destination  = os.path.join('uploaded_files')
        if not os.path.exists(destination):
            os.makedirs(destination)


        file_ = request.files.getlist("newfile")[0]
        filename = file_.filename
        dest = os.path.join('uploaded_files',filename)
        file_.save(dest)

        X = []
        Y_actual = []
        total = 0
        with open(dest,'r') as data:
            for line in data:
                total += 1
                temp = (line.split(','))
                if(len(temp) ==2):
                    Y_actual.append(temp[0])
                    X.append(temp[1].strip())
                else:
                    return render_template('data_error.html')

        results = clf.predict(X)

        probability = clf.predict_proba(X)
        confidence = []
        for i in probability:
            confidence.append(max(i))

        report = (classification_report(Y_actual,results)).split('\n')
        accuracy = np.mean(results == Y_actual)
        cnf_matrix = confusion_matrix(Y_actual, results)

        class_names = []
        for i in range(2,len(report)):
            class_names.append(''.join([i for i in report[i] if not i.isdigit()]).replace('.','').strip())

        new_cnf = []
        for i in range(len(cnf_matrix)):
            x = [class_names[i]]
            for item in cnf_matrix[i]:
                x.append(item)

            new_cnf.append(x)

            new__cnf = []
            for i in range(len(new_cnf)):
                new__cnf.append([])
                for j in range(len(new_cnf[i])):
                    if(type(new_cnf[i][j]) is not str):
                        new__cnf[i].append(float(new_cnf[i][j]))
                    else:
                        new__cnf[i].append(new_cnf[i][j])

        if(request.form.get('view_results')):
            return jsonify(report = report,accuracy=accuracy, total = len(X),confusion_matrix = new__cnf,results = results.tolist(),confidence = confidence)
        else:
            return jsonify(report = report,accuracy=accuracy, total = len(X),confusion_matrix = new__cnf, confidence = confidence)

    return render_template('uploaded_predict_accuracy.html')



@app.route('/return_dataset1')
def return_dataset1():
    filename = 'dataset_500'
    return send_file('datasets/'+filename+'.txt',mimetype='text/csv',attachment_filename = filename+'.txt',as_attachment=  True)

@app.route('/return_dataset2')
def return_dataset2():
    filename = 'dataset_1000'
    return send_file('datasets/'+filename+'.txt',mimetype='text/csv',attachment_filename = filename+'.txt',as_attachment=  True)

@app.route('/return_dataset3')
def return_dataset3():
    filename = 'dataset_5000'
    return send_file('datasets/'+filename+'.txt',mimetype='text/csv',attachment_filename = filename+'.txt',as_attachment=  True)

@app.route('/return_dataset4')
def return_dataset4():
    filename = 'dataset_10000'
    return send_file('datasets/'+filename+'.txt',mimetype='text/csv',attachment_filename = filename+'.txt',as_attachment=  True)

@app.route('/return_dataset5')
def return_dataset5():
    filename = 'dataset_25000'
    return send_file('datasets/'+filename+'.txt',mimetype='text/csv',attachment_filename = filename+'.txt',as_attachment=  True)


if __name__ == "__main__":
    app.run(debug = True)
