#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <iostream>
#include <fstream>
#include <string>

#include "math.h"
#include "dirent.h"
#include "face_detection.h"
#include "..\..\..\FaceAlignment\include\face_alignment.h"
#include "..\..\..\FaceIdentification\include\face_identification.h"

using namespace std;
using namespace seeta;
using namespace cv;

const string DETECTION_MODEL_DIR = "../../model/seeta_fd_frontal_v1.0.bin";
const string ALIGNMENT_MODEL_DIR = "../../../FaceAlignment/model/seeta_fa_v1.1.bin";
const string IDENTIFICATION_MODEL_DIR = "../../../FaceIdentification/model/seeta_fr_v1.0.bin";
const string LFW_DIR = "../../data/lfw/";

const string IMGS_SRC_DIR = "images/";
const string CROP_DIR = "crop/";
const string NO_DETECTED_DIR = "not_detected/";

const int FEAT_SIZE = 2048;

seeta::FaceDetection detector(DETECTION_MODEL_DIR.c_str());
seeta::FaceAlignment aligner(ALIGNMENT_MODEL_DIR.c_str());
seeta::FaceIdentification recognizer(IDENTIFICATION_MODEL_DIR.c_str());

struct Pair {
	string img_1;
	string img_2;
	float similarity;
	bool is_match;
};

struct FaceScore {
	int index;
	int score;
	double confidence;
	double distance;
	double size;
	cv::Rect bounding_box;
	string color;
};

// Calcula as acurácias da métrica de pares do LFW 
// para 1000 limiares, variando de 0.001 à 1.0.
// Os resultados são salvos no arquivo de destino e
// o limiar que representar a melhor acurácia é destacado
void CalcAcc(string similarities_file, string dst_file);

// Calcula a melhor pontuação entre os scores de face
// passados como parâmetro, comparando todos os scores
// e incrementando a pontuação dos face scores com menor
// distância para o centro da imagem, maior grau de confiança
// e dimensões
FaceScore CalcBestScore(FaceScore* scores, int length);

// Calcula os falsos negativos na métrica de pares do LFW 
// (faces de pessoas iguais mas obtiveram uma similaridade
// abaixo do limiar)
void CalcFalseNegatives(string similarities_file, string dst_file, float threshold);

// Calcula os falsos positivos na métrica de pares do LFW 
// (faces de pessoas diferentes mas obtiveram uma similaridade
// acima do limiar)
void CalcFalsePositives(string similarities_file, string dst_file, float threshold);

// Calcula os verdadeiros e falsos positivos, na métrica de
// pares do LFW, para 1000 limiares, variando de 0.001 à 1.0.
// Os resultados são salvos em arquivo e com ele pode-se gerar
// a curva ROC
void CalcRocCurve(string similarities_file, string dst_file);

// Calcula a similaridade dos pares na métrica de pares do LFW
// e salva em arquivo em formato "img1 img2 similarity_val"
// (sem aspas)
void CalcSimilarities(string pairs_file, string similarities_file);

// ?
void CalcSimilaritiesForCorrectedFaces(string src_file, string old_dst_file, string new_dst_file);

// Calcula a similaridade das imagens que não obtiveram
// faces detectadas (sim=0.0). Antes as imagens devem ter
// sido cropadas, usando landmarks manualmente localizados
void CalcSimilatiriesForNoCroppedImgs(string src_file);

// Copia imagens em pares do arquivo de pares do LFW
// para uma pasta destino
void CopyImgs(string src_file, string dst_file);
 
// Copia as imagens que não obtiveram faces encontradas
// para uma pasta de destino
void CopyNoCroppedImgs(string src_file, string dst_file);

// Corta uma imagem usando landmarks de face como 
// delimitadores e salva em uma pasta
void Crop(Mat img, FacialLandmark points[5], string img_name);

// Corta todas as imagens especificadas num arquivo
// de texto, usando landmarks de face como delimitadores
// e informações de distancia para a região central,
// grau de confiabilidade de faces e dimensões das faces
// detectadas, construindo uma pontuação usando tais infos 
// e salvando as melhores faces cortadas
void CropAdvanced(string src_file, string dst_file);

// Corta uma única imagem, usando landmarks de face como delimitadores
// e informações de distancia para a região central,
// grau de confiabilidade de faces e dimensões das faces
// detectadas, construindo uma pontuação usando tais infos 
// e salvando a melhor face cortada para ser usada na métrica
// de pares do LFW
void CropImgAdvanced(string src_file, string dst_file);

// Corta todas as imagens especificadas num arquivo
// de texto, usando landmarks de face como delimitadores
// e a informação de distancia para a região central,
// salvando as melhores faces cortadas
void CropWithCenterRegion(string src_file, string dst_file);

// Corta todas as imagens especificadas num arquivo
// de texto, usando landmarks de face como delimitadores
// e salvando as imagens em uma pasta
void CropWithManuallyExtractedLandmarks(string src_file, string feats_file);

// Extrai e salva as features de todas as imagens
// da lista txt de imagens em um arquivo de destino
void SaveFeats(string lfw_list_file, string feats_file);

// Compara a lista de imagens originais com a lista
// de imagens cortadas e preenche um arquivo com os
// nomes das imagens que NÃO foram cortadas
void SetNoCroppedImgsList(string dst_file);

// Realiza a detecção de faces de uma imagem
// utilizando os atributos passados como parâmetros
void TestFaceDetection(string src_file, int32_t min_size, float thresh, float factor, int32_t setp_x, int32_t setp_y);

// Realiza a detecção de faces de uma imagem
// utilizando diversas variações dos parâmetros
void TestFaceDetectionWithAllParams(string src_file_name, string dst_file);

// Calcula a distância euclideana entre dois pontos 2D
double CalcEuclideanDistance(Point p1, Point p2);

// Calcula o tempo médio de extração de features na
// métrica de pares do LFW
double CalcProcessingMeanTime(string pairs_file);

// Extrai as características de uma imagem com face
// detectada e retorna um array
float* ExtractFeats(string img_file);

// Recupera as features de uma imagem específica
// salvas em um arquivo
float* GetFeats(string img_file, string feats_file);

// Converte uma imagem tipo Mat, do OpenCV, para um
// uma imagem tipo ImageData em escala de cinza
ImageData ConvertMatImageToGrayImageData(Mat img);

// Forma o nome original da imagem usando as combinações
// do arquivo de pares do LFW, concatenando o nome da pessoa
// ao número
string SetImgFileName(string img_file_name, int number);

// Extrai as características de uma imagem com face
// e retorna uma string com o formato salvo no arquivo
string ExtractFeats(string person_name, string img_file);

// Converte string em wstring (usado para criar
// diretório de imagens)
wstring s2ws(const std::string& s);

int main(int argc, char** argv) {
	//CropWithManuallyExtractedLandmarks("not_crop.txt", "feats.txt");
	//CalcSimilatiriesForNoCroppedImgs("pairs-similarities.txt");
	//CalcSimilarities("pairs.txt", "pairs-similarities_02.txt");
	//SetNoCroppedImgsList("non_cropped.txt");
	//CalcRocCurve("pairs-similarities_02.txt", "ROC-LFW-SeetaFace_02.txt");
	//CalcAcc("pairs-similarities_02.txt", "ACC-LFW-SeetaFace_02.txt");
	//CalcFalsePositives("pairs-similarities_02.txt", "false-positives_02.txt", 0.442998);
	//CalcFalseNegatives("pairs-similarities_02.txt", "false-negatives_02.txt", 0.442998);
	//CopyImgs("false-negatives_02.txt", "false_negatives_02");
	//CopyImgs("false-positives_02.txt", "false_positives_02");
	//CopyNoCroppedImgs("no_cropped.txt", "not_detected");
	//TestFaceDetection("EXCLUIR.jpg", 40, 2.f, 0.8f, 4, 4);
	//TestFaceDetectionWithAllParams("Mohammad_Fares_0001.jpg");
	//CropWithCenterRegion("both-face-detection-error.txt", "fd_center_region_correction");
	//CropAdvanced("false-negatives-face-detection-error.txt", "fd_advanced_no_cr_error");
	//CropImgAdvanced("Jeanne_Moreau_0001.jpg", "face_detection_failures");
	//CalcSimilaritiesForCorrectedFaces("face_detection_failures.txt", "pairs-similarities.txt", "pairs-similarities_02.txt");
	CalcProcessingMeanTime("pairs.txt");
}

void CalcAcc(string similarities_file, string dst_file) {
	// ACC = TP + TN / P + N
	cout << "Starting accuracy calculation..." << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + similarities_file, ifstream::in);

	filebuf fb;
	fb.open(LFW_DIR + dst_file, ios_base::app);
	ostream os(&fb);

	string line;
	int num_lines = 0;

	while (getline(ifs, line))
		num_lines++;

	ifs.close();
	ifs.open(LFW_DIR + similarities_file, ifstream::in);

	Pair* pairs = new Pair[num_lines];

	string word;
	int offset = 0;
	int index = 0;
	bool is_match = true;

	while (ifs >> word) {
		if (offset == 300) {
			is_match = !is_match;
			offset = 0;
		}

		pairs[index].img_1 = word;
		ifs >> word;
		pairs[index].img_2 = word;
		ifs >> word;
		pairs[index].similarity = strtof((word).c_str(), 0);

		pairs[index].is_match = is_match ? true : false;

		offset++;
		index++;
	}

	float threshold = 0.0f;
	int p = 0;
	int n = 0;

	for (int i = 0; i < num_lines; i++)
		pairs[i].is_match ? p++ : n++;

	cout << "Calculating accuracy for 1000 thresholds..." << endl;

	float high_acc = 0.0f;
	float selected_threshold = threshold;

	while (threshold < 1.0f) {
		float tp = 0.0f;
		float tn = 0.0f;
		float acc = 0.0f;
		for (int i = 0; i < num_lines; i++) {
			if (pairs[i].similarity >= threshold && pairs[i].is_match)
				tp++;
			else if (pairs[i].similarity < threshold && !pairs[i].is_match)
				tn++;
		}

		acc = (tp + tn) / (p + n);

		if (acc > high_acc) {
			high_acc = acc;
			selected_threshold = threshold;
		}

		cout << threshold << " " << acc << endl;
		os << threshold << " " << acc << "\n";

		threshold += 0.001;
	}

	os << "\n\n" << selected_threshold << " " << high_acc << endl;

	ifs.close();
	fb.close();

	cout << "\n" << "Best accuracy: " << high_acc << endl;
	cout << "\n" << "Threshold: " << selected_threshold << endl;

	cout << "Successful!" << endl;
	system("pause");
}

FaceScore CalcBestScore(FaceScore* scores, int length) {
	FaceScore best_score;

	for (int i = 0; i < length; i++) {
		for (int k = i + 1; k < length; k++) {
			if (scores[i].distance < scores[k].distance)
				scores[i].score++;
			else if(scores[i].distance > scores[k].distance)
				scores[k].score++;

			if (scores[i].confidence > scores[k].confidence)
				scores[i].score++;
			else if(scores[i].confidence < scores[k].confidence)
				scores[k].score++;

			if (scores[i].size > scores[k].size)
				scores[i].score++;
			else if (scores[i].size < scores[k].size)
				scores[k].score++;
		}
	}

	best_score = scores[0];

	for (int i = 0; i < length; i++) {
		cout << "---------- Face " << (i + 1) << " ----------" << endl;
		cout << "distance: " << scores[i].distance << endl;
		cout << "confidence: " << scores[i].confidence << endl;
		cout << "size: " << scores[i].size << endl;
		cout << "color: " << scores[i].color << endl;
		cout << "SCORE: " << scores[i].score << endl;
		if (scores[i].score > best_score.score)
			best_score = scores[i];
	}

	return best_score;
}

void CalcFalseNegatives(string similarities_file, string dst_file, float threshold) {
	// 0.442998
	cout << "Calculating false negatives..." << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + similarities_file, ifstream::in);

	filebuf fb;
	fb.open(LFW_DIR + dst_file, ios_base::app);
	ostream os(&fb);

	string line;
	int num_lines = 0;

	while (getline(ifs, line))
		num_lines++;

	ifs.close();
	ifs.open(LFW_DIR + similarities_file, ifstream::in);

	Pair* pairs = new Pair[num_lines];

	string word;
	int offset = 0;
	int index = 0;
	bool is_match = true;

	while (ifs >> word) {
		if (offset == 300) {
			is_match = !is_match;
			offset = 0;
		}

		pairs[index].img_1 = word;
		ifs >> word;
		pairs[index].img_2 = word;
		ifs >> word;
		pairs[index].similarity = strtof((word).c_str(), 0);

		pairs[index].is_match = is_match ? true : false;

		offset++;
		index++;
	}

	cout << "Counting false negatives (" << "threshold = " << threshold << ")..." << endl;

	int fnc = 0;

	for (int i = 0; i < num_lines; i++) {
		if (pairs[i].similarity < threshold && pairs[i].is_match)
		{
			fnc++;
			cout << pairs[i].img_1 << " " << pairs[i].img_2 << " " << pairs[i].similarity << endl;
			os << pairs[i].img_1 << " " << pairs[i].img_2 << " " << pairs[i].similarity << endl;
		}
	}

	cout << fnc << endl;
	os << fnc << endl;

	ifs.close();
	fb.close();

	cout << "Successful!" << endl;
	system("pause");
}

void CalcFalsePositives(string pairs_similarities_file_name, string false_positives_file_name, float threshold) {
	// 0.442998
	cout << "Calculating false positives..." << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + pairs_similarities_file_name, ifstream::in);

	filebuf fb;
	fb.open(LFW_DIR + false_positives_file_name, ios_base::app);
	ostream os(&fb);

	string line;
	int num_lines = 0;

	while (getline(ifs, line))
		num_lines++;

	ifs.close();
	ifs.open(LFW_DIR + pairs_similarities_file_name, ifstream::in);

	Pair* pairs = new Pair[num_lines];

	string value;
	int offset = 0;
	int index = 0;
	bool is_match = true;

	while (ifs >> value) {
		if (offset == 300) {
			is_match = !is_match;
			offset = 0;
		}

		pairs[index].img_1 = value;
		ifs >> value;
		pairs[index].img_2 = value;
		ifs >> value;
		pairs[index].similarity = strtof((value).c_str(), 0);

		pairs[index].is_match = is_match ? true : false;

		offset++;
		index++;
	}

	cout << "Counting false positives (" << "threshold = " << threshold << ")..." << endl;

	int fpc = 0;

	for (int i = 0; i < num_lines; i++) {
		if (pairs[i].similarity >= threshold && !pairs[i].is_match)
		{
			fpc++;
			cout << pairs[i].img_1 << " " << pairs[i].img_2 << " " << pairs[i].similarity << endl;
			os << pairs[i].img_1 << " " << pairs[i].img_2 << " " << pairs[i].similarity << endl;
		}
	}

	cout << fpc << endl;
	os << fpc << endl;

	ifs.close();
	fb.close();

	cout << "Successful!" << endl;
	system("pause");
}

void CalcRocCurve(string similarities_file, string dst_file) {
	// TPR = TP / TP + FN or TPR = TP / P
	// FPR = FP / FP + TN or FPR = FP / N
	cout << "Starting roc curve calculation..." << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + similarities_file, ifstream::in);

	filebuf fb;
	fb.open(LFW_DIR + dst_file, ios_base::app);
	ostream os(&fb);

	string line;
	int num_lines = 0;

	while (getline(ifs, line))
		num_lines++;

	ifs.close();
	ifs.open(LFW_DIR + similarities_file, ifstream::in);

	Pair* pairs = new Pair[num_lines];

	string word;
	int offset = 0;
	int index = 0;
	bool is_match = true;

	cout << "Setting pairs data... " << endl;

	while (ifs >> word) {
		if (offset == 300) {
			is_match = !is_match;
			offset = 0;
		}

		pairs[index].img_1 = word;
		ifs >> word;
		pairs[index].img_2 = word;
		ifs >> word;
		pairs[index].similarity = strtof((word).c_str(), 0);

		pairs[index].is_match = is_match ? true : false;

		offset++;
		index++;
	}

	float threshold = 0.0f;
	int p = 0;
	int n = 0;

	for (int i = 0; i < num_lines; i++)
		pairs[i].is_match ? p++ : n++;

	cout << "Calculating TPF and FPR for 1000 thresholds..." << endl;

	while (threshold < 1.0f) {
		float tp = 0.0f;
		float fp = 0.0f;
		for (int i = 0; i < num_lines; i++) {
			if (pairs[i].similarity >= threshold && pairs[i].is_match)
				tp++;
			else if (pairs[i].similarity >= threshold && !pairs[i].is_match)
				fp++;
		}

		double tpr = tp / p;
		double fpr = fp / n;

		cout << tpr << " " << fpr << endl;
		os << tpr << " " << fpr << "\n";

		threshold += 0.001;
	}

	ifs.close();
	fb.close();

	cout << "Successful!" << endl;
	system("pause");
}

void CalcSimilarities(string pairs_file, string similarities_file) {
	cout << "Starting similarities calculation..." << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + pairs_file, ifstream::in);

	filebuf fb;
	fb.open(LFW_DIR + similarities_file, ios_base::app);
	ostream os(&fb);

	int k = 0;
	int n = 0;

	ifs >> k;
	ifs >> n;

	int offset = 0;
	bool flag = false;

	while (!ifs.eof()) {
		string person_name;
		string person_name_2;
		string img_file_1;
		string img_file_2;
		float *img_feats_1;
		float *img_feats_2;
		int n1;
		int n2;

		if (offset == 300) {
			flag = !flag;
			offset = 0;
		}

		if (!flag) {
			ifs >> person_name;
			ifs >> n1;
			ifs >> n2;

			cout << "========== Extracting " << person_name << "'s features... ==========" << endl;

			img_file_1 = SetImgFileName(person_name, n1);
			img_feats_1 = ExtractFeats(img_file_1);
			img_file_2 = SetImgFileName(person_name, n2);
			img_feats_2 = ExtractFeats(img_file_2);
		}
		else {
			ifs >> person_name;
			ifs >> n1;
			ifs >> person_name_2;
			ifs >> n2;

			cout << "========== Extracting " << person_name << " and " << person_name_2 << "'s features... ==========" << endl;

			img_file_1 = SetImgFileName(person_name, n1);
			img_feats_1 = ExtractFeats(img_file_1);
			img_file_2 = SetImgFileName(person_name_2, n2);
			img_feats_2 = ExtractFeats(img_file_2);
		}

		float sim = 0.0;

		if (img_feats_1 == NULL || img_feats_2 == NULL)
			//os << img_file_1 << " " << img_file_2 << "0.0" << endl;
			;
		else {
			cout << "Calculating similarity..." << endl;
			sim = recognizer.CalcSimilarity(img_feats_1, img_feats_2);
			//os << img_file_1 << " " << img_file_2 << " " << sim << endl;
		}

		cout << "sim: " + to_string(sim) << endl;

		offset++;
	}

	ifs.close();
	fb.close();

	cout << "\nSuccessful!" << endl;
	system("pause");
}

void CalcSimilaritiesForCorrectedFaces(string src_file, string old_dst_file, string new_dst_file) {
	ifstream src_ifs;
	src_ifs.open(LFW_DIR + src_file, ifstream::in);

	ifstream old_dst_ifs;
	old_dst_ifs.open(LFW_DIR + old_dst_file, ifstream::in);

	filebuf fb;
	fb.open(LFW_DIR + new_dst_file, ios_base::app);
	ostream os(&fb);

	int num_lines = 0;

	string img_file;

	while (src_ifs >> img_file) {
		src_ifs >> img_file;
		num_lines++;
	}

	string* img_files = new string[num_lines];
	src_ifs.close();
	src_ifs.open(LFW_DIR + src_file, ifstream::in);

	for (int i = 0; i < num_lines; i++)
		src_ifs >> img_files[i] >> img_files[i];
	src_ifs.close();
	
	string img_file_1;
	string img_file_2;
	string sim;
	bool flag = false;
	while (old_dst_ifs >> img_file_1) {	
		old_dst_ifs >> img_file_2;
		old_dst_ifs >> sim;
		flag = false;
		cout << "========== " << img_file_1 << " x " << img_file_2 << " ==========" << endl;
		for (int i = 0; i < num_lines; i++) {
			if (img_file_1.compare(img_files[i]) == 0 || img_file_2.compare(img_files[i]) == 0) {
				float* img_feats_1 = ExtractFeats(img_file_1);
				float* img_feats_2 = ExtractFeats(img_file_2);

				float new_sim = 0.0f;

				string new_line = img_file_1 + " " + img_file_2 + " " + to_string(new_sim);

				if (img_feats_1 == NULL || img_feats_2 == NULL) {
					os << new_line << endl;
				}
				else {
					cout << "Calculating similarity..." << endl;
					new_sim = recognizer.CalcSimilarity(img_feats_1, img_feats_2);
					new_line = img_file_1 + " " + img_file_2 + " " + to_string(new_sim);
					cout << "New similarity: " << to_string(new_sim) << endl;
					os << new_line << endl;
				}

				delete[] img_feats_1;
				delete[] img_feats_2;

				flag = true;

				break;
			}
		}
		
		if (!flag)
		{
			os << img_file_1 << " " << img_file_2 << " " << sim << endl;
			cout << "sim: " << sim << endl;
		}
			
	}


	fb.close();
	old_dst_ifs.close();
	delete[] img_files;

	cout << "Successful!" << endl;
	system("pause");
}

void CalcSimilatiriesForNoCroppedImgs(string src_file) {
	cout << "Starting similarities calculation for no cropped images..." << endl;

	fstream fs;
	fs.open(LFW_DIR + src_file);

	int curr_line = 1;

	while (!fs.eof()) {
		string sim_str;
		string img_file_1;
		string img_file_2;
		float sim = 0.0f;

		cout << "===== " << img_file_1 << " x " << img_file_2 << " =====" << endl;

		try
		{
			fs >> img_file_1 >> img_file_2 >> sim_str;
			sim = strtof((sim_str).c_str(), 0);
		}
		catch (exception e)
		{
			break;
		}

		if (sim == 0.0f) {
			cout << "Extracting features..." << endl;
			float* img_feats_1 = ExtractFeats(img_file_1);
			float* img_feats_2 = ExtractFeats(img_file_2);

			if (img_feats_1 == NULL || img_feats_2 == NULL) {
				cout << "Features extraction error has ocurred in line " << curr_line << "\n" << endl;
			}
			else {
				cout << "Features extracted!" << endl;
				cout << "Calculating similarity..." << endl;

				sim = recognizer.CalcSimilarity(img_feats_1, img_feats_2);

				string line = img_file_1 + " " + img_file_2 + " " + "0.0";

				fs.seekp(-ios::off_type(line.size()) - 1, ios_base::cur);
				fs << img_file_1 << " " << img_file_2 << " " << sim << endl;

				cout << "Well done!" << endl;
			}
		}
		cout << "==============================" << endl;

		curr_line++;
	}

	fs.close();

	cout << "Successful!" << endl;
	system("pause");
}

void CopyImgs(string src_file, string dst_file) {
	cout << "Starting images copying..." << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + src_file, ifstream::in);

	std::wstring stemp = s2ws(LFW_DIR + dst_file);
	LPCWSTR result = stemp.c_str();

	if (!CreateDirectory(result, NULL))
		cout << LFW_DIR + dst_file << " already exists!" << endl;

	int cont = 0;
	string word;

	while (ifs >> word) {
		string img_1 = word;
		ifs >> word;
		string img_2 = word;
		ifs >> word;

		DIR *dir;
		struct dirent *ent;

		string cropped_imgs_path = LFW_DIR + CROP_DIR;
		const char *cropped_imgs_dir = (cropped_imgs_path).c_str();

		int copies = 0;

		if ((dir = opendir(cropped_imgs_dir)) != NULL) {
			Mat img_1_copy;
			string img_1_name;
			Mat img_2_copy;
			string img_2_name;
			while ((ent = readdir(dir)) != NULL) {
				std::string cropped_img_name(ent->d_name);
				if (cropped_img_name.compare(".") != 0 && cropped_img_name.compare("..") != 0) {
					if (cropped_img_name.compare(img_1) == 0) {
						img_1_name = cropped_img_name;
						img_1_copy = imread(cropped_imgs_path + cropped_img_name, 1);
						copies++;
					}

					if (cropped_img_name.compare(img_2) == 0) {
						img_2_name = cropped_img_name;
						img_2_copy = imread(cropped_imgs_path + cropped_img_name, 1);
						copies++;
					}

					if (copies == 2) {
						cont++;
						string dst_path = LFW_DIR + dst_file + "/" + to_string(cont);
						wstring stemp = s2ws(dst_path);
						LPCWSTR result = stemp.c_str();

						if (!CreateDirectory(result, NULL))
							cout << dst_path << " already exists!" << endl;

						imwrite(dst_path + "/" + img_1_name, img_1_copy);
						imwrite(dst_path + "/" + img_2_name, img_2_copy);

						break;
					}
				}
			}
		}
	}

	ifs.close();
	cout << "Successful!" << endl;
	system("pause");
}

void CopyNoCroppedImgs(string src_file, string dst_file) {
	cout << "Starting images copying..." << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + src_file, ifstream::in);

	std::wstring stemp = s2ws(LFW_DIR + dst_file);
	LPCWSTR result = stemp.c_str();

	if (!CreateDirectory(result, NULL))
		cout << LFW_DIR + dst_file << " already exists!" << endl;

	int cont = 0;

	string word;
	while (ifs >> word) {
		string img = word;

		DIR *dir;
		struct dirent *ent;

		string cropped_imgs_path = LFW_DIR + CROP_DIR;
		const char *cropped_imgs_dir = (cropped_imgs_path).c_str();

		if ((dir = opendir(cropped_imgs_dir)) != NULL) {
			Mat img_copy;
			string img_name;
			while ((ent = readdir(dir)) != NULL) {
				std::string cropped_img_name(ent->d_name);
				if (cropped_img_name.compare(".") != 0 && cropped_img_name.compare("..") != 0) {
					if (cropped_img_name.compare(img) == 0) {
						img_name = cropped_img_name;
						img_copy = imread(cropped_imgs_path + cropped_img_name, 1);
						string dst_path = LFW_DIR + dst_file;
						wstring stemp = s2ws(dst_path);
						LPCWSTR result = stemp.c_str();

						if (!CreateDirectory(result, NULL))
							cout << dst_path << " already exists!" << endl;

						imwrite(dst_path + "/" + img_name, img_copy);

						break;
					}
				}
			}
		}
	}

	ifs.close();
	cout << "Successful!" << endl;
	system("pause");
}

void Crop(Mat img, FacialLandmark points[5], string img_name) {
	ImageData img_data;
	img_data.data = img.data;
	img_data.width = img.cols;
	img_data.height = img.rows;
	img_data.num_channels = img.channels();

	Mat dst_img(recognizer.crop_height(), recognizer.crop_width(), CV_8UC(recognizer.crop_channels()));

	ImageData cropped_image;
	cropped_image.data = dst_img.data;
	cropped_image.width = dst_img.cols;
	cropped_image.height = dst_img.rows;
	cropped_image.num_channels = dst_img.channels();

	recognizer.CropFace(img_data, points, cropped_image);

	string cropped_img_path = LFW_DIR + CROP_DIR + img_name;
	imwrite(cropped_img_path, dst_img);
}

void CropAdvanced(string src_file, string dst_file) {
	cout << "Starting images cropping advanced...\n" << endl;

	detector.SetMinFaceSize(40); // size >= 20
	detector.SetScoreThresh(2.f); // default is 2.0 (could be 0.95, 2.8 or 4.5)
	detector.SetImagePyramidScaleFactor(0.8f); // 0.7, 0.8 or 1.25
	detector.SetWindowStep(4, 4); // 2 or 4

	ifstream ifs;
	ifs.open(LFW_DIR + src_file, ifstream::in);
	string word;

	while (ifs >> word) {
		string file_number = word;
		string img_name;
		ifs >> img_name;

		DIR *dir;
		struct dirent *ent;

		string imgs_path = LFW_DIR + IMGS_SRC_DIR;
		const char *imgs_dir = (imgs_path).c_str();

		if ((dir = opendir(imgs_dir)) != NULL) {
			while ((ent = readdir(dir)) != NULL) {
				string sub_dir_name(ent->d_name);
				if (sub_dir_name.compare(".") != 0 && sub_dir_name.compare("..") != 0) {
					DIR *sub_dir;
					struct dirent *sub_ent;

					string imgs_sub_path = imgs_path + sub_dir_name;
					const char *imgs_sub_dir = (imgs_sub_path).c_str();

					if ((sub_dir = opendir(imgs_sub_dir)) != NULL) {
						while ((sub_ent = readdir(sub_dir)) != NULL) {
							string img_name_to_comp(sub_ent->d_name);

							if (img_name_to_comp.compare(".") != 0 && img_name_to_comp.compare("..") != 0) {
								if (img_name_to_comp.compare(img_name) == 0) {
									cout << "========== " << img_name << " ==========" << endl;

									string img_final_path = imgs_sub_path + "/" + img_name;
									Mat img = imread(img_final_path, 1);

									if (!img.empty()) {
										Mat gray_img;

										if (img.channels() != 1)
											cvtColor(img, gray_img, COLOR_BGR2GRAY);
										else
											gray_img = img;

										seeta::ImageData gray_img_data;
										gray_img_data.data = gray_img.data;
										gray_img_data.width = gray_img.cols;
										gray_img_data.height = gray_img.rows;
										gray_img_data.num_channels = 1;

										cout << "Detecting faces..." << endl;
										vector<FaceInfo> faces = detector.Detect(gray_img_data);

										int32_t num_faces = static_cast<int32_t>(faces.size());
										Mat img_copy = img.clone();

										if (num_faces > 0) {
											cout << num_faces << " detected!" << endl;
											cout << "Searching for the correct face..." << endl;

											Point img_center = Point(img_copy.cols / 2, img_copy.rows / 2);
											cout << "Image center: (" << img_center.x << ", " << img_center.y << ")" << endl;

											FaceScore* face_scores = new FaceScore[num_faces];

											for (int32_t i = 0; i < num_faces; i++) {
												cv::Rect curr_face_rect;
												curr_face_rect.x = faces[i].bbox.x;
												curr_face_rect.y = faces[i].bbox.y;
												curr_face_rect.width = faces[i].bbox.width;
												curr_face_rect.height = faces[i].bbox.height;

												Point face_center = Point(curr_face_rect.x + curr_face_rect.width / 2,
													curr_face_rect.y + curr_face_rect.height / 2);

												int face_num = i + 1;

												//cout << "---------- Face " << face_num << " ----------" << endl;
												//cout << "Center: (" << face_center.x << ", " << face_center.y << ")" << endl;

												double distance = cv::norm(img_center - face_center);
												//cout << "Distance to the center of the image: " << distance << endl;

												double confidence = faces[i].score;
												//cout << "Confidence score: " << confidence << endl;

												double size = faces[i].bbox.width * faces[i].bbox.height;
												//cout << "Size: " << size << endl;

												face_scores[i].index = i;
												face_scores[i].score = 0;
												face_scores[i].distance = distance;
												face_scores[i].size = size;
												face_scores[i].confidence = confidence;
												face_scores[i].bounding_box = curr_face_rect;

												if (i == 1) 
													face_scores[i].color = "green";
												else if (i > 1) {
													face_scores[i].color = "blue";
												}
												else
													face_scores[i].color = "red";
											}

											FaceScore best_score = CalcBestScore(face_scores, num_faces);

											cout << "\nBEST SCORE: Face " << (best_score.index + 1) << endl;
											cout << "Detecting landmarks..." << endl;

											FacialLandmark points[5];
											bool is_there_landmarks = aligner.PointDetectLandmarks(gray_img_data, faces[best_score.index], points);

											if (is_there_landmarks) {
												cout << "Cropping..." << endl;

												Crop(img, points, img_name);

												cout << "Successful!" << endl;
											}
											else
												cout << "No landmarks detected...";

											rectangle(img_copy, best_score.bounding_box, CV_RGB(0, 0, 255), 4, 8, 0);
											imwrite(LFW_DIR + dst_file + "/" + img_name, img_copy);
											delete[] face_scores;
										}
										else
											cout << "No faces detected!" << endl;

									}
								}
							}
						}
					}
				}
			}
		}
	}

	ifs.close();

	system("pause");
}

void CropImgAdvanced(string src_file, string dst_file) {
	cout << "Starting image cropping advanced...\n" << endl;

	detector.SetMinFaceSize(40); // size >= 20
	detector.SetScoreThresh(2.f); // default is 2.0 (could be 0.95, 2.8 or 4.5)
	detector.SetImagePyramidScaleFactor(0.8f); // 0.7, 0.8 or 1.25
	detector.SetWindowStep(2, 2); // 2 or 4

	DIR *dir;
	struct dirent *ent;

	string imgs_path = LFW_DIR + IMGS_SRC_DIR;
	const char *imgs_dir = (imgs_path).c_str();

	if ((dir = opendir(imgs_dir)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			string sub_dir_name(ent->d_name);
			if (sub_dir_name.compare(".") != 0 && sub_dir_name.compare("..") != 0) {
				DIR *sub_dir;
				struct dirent *sub_ent;

				string imgs_sub_path = imgs_path + sub_dir_name;
				const char *imgs_sub_dir = (imgs_sub_path).c_str();

				if ((sub_dir = opendir(imgs_sub_dir)) != NULL) {
					while ((sub_ent = readdir(sub_dir)) != NULL) {
						string img_name_to_comp(sub_ent->d_name);

						if (img_name_to_comp.compare(".") != 0 && img_name_to_comp.compare("..") != 0) {
							if (img_name_to_comp.compare(src_file) == 0) {
								cout << "========== " << src_file << " ==========" << endl;

								string img_final_path = imgs_sub_path + "/" + src_file;
								Mat img = imread(img_final_path, 1);

								if (!img.empty()) {
									Mat gray_img;

									if (img.channels() != 1)
										cvtColor(img, gray_img, COLOR_BGR2GRAY);
									else
										gray_img = img;

									seeta::ImageData gray_img_data;
									gray_img_data.data = gray_img.data;
									gray_img_data.width = gray_img.cols;
									gray_img_data.height = gray_img.rows;
									gray_img_data.num_channels = 1;

									cout << "Detecting faces..." << endl;
									vector<FaceInfo> faces = detector.Detect(gray_img_data);

									int32_t num_faces = static_cast<int32_t>(faces.size());
									Mat img_copy = img.clone();

									if (num_faces > 0) {
										cout << num_faces << " detected!" << endl;
										cout << "Searching for the correct face..." << endl;

										Point img_center = Point((img_copy.cols) / 2, (img_copy.rows) / 2);
										cout << "Image center: (" << img_center.x << ", " << img_center.y << ")" << endl;

										FaceScore* face_scores = new FaceScore[num_faces];
										int length = 0;

										int r = 255;
										int g = 0;
										int b = 0;

										for (int32_t i = 0; i < num_faces; i++) {
											cv::Rect curr_face_rect;
											curr_face_rect.x = faces[i].bbox.x;
											curr_face_rect.y = faces[i].bbox.y;
											curr_face_rect.width = faces[i].bbox.width;
											curr_face_rect.height = faces[i].bbox.height;

											Point face_center = Point(faces[i].bbox.x + faces[i].bbox.width / 2,
												faces[i].bbox.y + faces[i].bbox.height / 2);

											int face_num = i + 1;

											//cout << "---------- Face " << face_num << " ----------" << endl;
											//cout << "Center: (" << face_center.x << ", " << face_center.y << ")" << endl;

											double distance = cv::norm(img_center - face_center);
											//cout << "Distance to the center of the image: " << distance << endl;

											double confidence = faces[i].score;
											//cout << "Confidence score: " << confidence << endl;

											double size = faces[i].bbox.width * faces[i].bbox.height;
											//cout << "Size: " << size << endl;

											face_scores[i].index = i;
											face_scores[i].score = 0;
											face_scores[i].distance = distance;
											face_scores[i].size = size;
											face_scores[i].confidence = confidence;
											face_scores[i].bounding_box = curr_face_rect;
											length++;

											if (i == 1) {
												r = 0;
												g = 255;
												b = 0;
												face_scores[i].color = "green";
											}
											else if (i > 1) {
												r = 0;
												g = 0;
												b = 255;
												face_scores[i].color = "blue";
											}
											else 
												face_scores[i].color = "red";

											rectangle(img_copy, curr_face_rect, CV_RGB(b, g, r), 4, 8, 0);
										}

										FaceScore best_score = CalcBestScore(face_scores, length);

										cout << "\nBEST SCORE: Face " << (best_score.index + 1) << endl;
										cout << "Detecting landmarks..." << endl;

										FacialLandmark points[5];
										bool is_there_landmarks = aligner.PointDetectLandmarks(gray_img_data, faces[best_score.index], points);

										if (is_there_landmarks) {
											cout << "Cropping..." << endl;

											Crop(img, points, src_file);

											cout << "Successful!" << endl;
										}
										else
										cout << "No landmarks detected...";

										imwrite(LFW_DIR + dst_file + "/" + src_file, img_copy);
										delete[] face_scores;
									}
									else
										cout << "No faces detected!" << endl;
								}
							}
						}
					}
				}
			}
		}
	}
	
	system("pause");
}

void CropWithCenterRegion(string src_file, string dst_file) {
	cout << "Starting images cropping using center region...\n" << endl;

	detector.SetMinFaceSize(40); // size >= 20
	detector.SetScoreThresh(0.95f); // default is 2.0 (could be 0.95, 2.8 or 4.5)
	detector.SetImagePyramidScaleFactor(0.8f); // 0.7, 0.8 or 1.25
	detector.SetWindowStep(4, 4); // 2 or 4

	ifstream ifs;
	ifs.open(LFW_DIR + src_file, ifstream::in);
	string word;

	while (ifs >> word) {
		string file_number = word;
		string img_name;
		ifs >> img_name;

		DIR *dir;
		struct dirent *ent;

		string imgs_path = LFW_DIR + "images/";
		const char *imgs_dir = (imgs_path).c_str();

		if ((dir = opendir(imgs_dir)) != NULL) {
			while ((ent = readdir(dir)) != NULL) {
				string sub_dir_name(ent->d_name);
				if (sub_dir_name.compare(".") != 0 && sub_dir_name.compare("..") != 0) {
					DIR *sub_dir;
					struct dirent *sub_ent;

					string imgs_sub_path = imgs_path + sub_dir_name;
					const char *imgs_sub_dir = (imgs_sub_path).c_str();

					if ((sub_dir = opendir(imgs_sub_dir)) != NULL) {
						while ((sub_ent = readdir(sub_dir)) != NULL) {
							string img_name_to_comp(sub_ent->d_name);

							if (img_name_to_comp.compare(".") != 0 && img_name_to_comp.compare("..") != 0) {
								if (img_name_to_comp.compare(img_name) == 0) {
									cout << "========== " << img_name << " ==========" << endl;

									string img_final_path = imgs_sub_path + "/" + img_name;
									Mat img = imread(img_final_path, 1);

									if (!img.empty()) {
										Mat gray_img;

										if (img.channels() != 1)
											cvtColor(img, gray_img, COLOR_BGR2GRAY);
										else
											gray_img = img;

										seeta::ImageData gray_img_data;
										gray_img_data.data = gray_img.data;
										gray_img_data.width = gray_img.cols;
										gray_img_data.height = gray_img.rows;
										gray_img_data.num_channels = 1;

										cout << "Detecting faces..." << endl;
										vector<FaceInfo> faces = detector.Detect(gray_img_data);

										int32_t num_faces = static_cast<int32_t>(faces.size());
										Mat img_copy = img.clone();

										double lowest_distance = 0.0;
										int selected_face_index = 0;
										cv::Rect selected_face_bounding_box;

										if (num_faces > 0) {
											cout << num_faces << " detected!" << endl;
											cout << "Searching for the correct face..." << endl;

											Point img_center = Point((img_copy.cols) / 2, (img_copy.rows) / 2);
											cout << "Image center: (" << img_center.x << ", " << img_center.y << ")" << endl;

											for (int32_t i = 0; i < num_faces; i++) {
												cv::Rect curr_face_rect;
												curr_face_rect.x = faces[i].bbox.x;
												curr_face_rect.y = faces[i].bbox.y;
												curr_face_rect.width = faces[i].bbox.width;
												curr_face_rect.height = faces[i].bbox.height;

												Point face_center = Point(curr_face_rect.x + curr_face_rect.width / 2,
													curr_face_rect.y + curr_face_rect.height / 2);

												int face_num = i + 1;

												cout << "---------- Face " << face_num << " ----------" << endl;
												cout << "Center: (" << face_center.x << ", " << face_center.y << ")" << endl;

												double distance = cv::norm(img_center - face_center);
												cout << "Distance to the center of the image: " << distance << endl;
												
												if (i == 0) {
													selected_face_index = i;
													lowest_distance = distance;
													selected_face_bounding_box = curr_face_rect;
												}
												else {
													if (distance < lowest_distance) {
														lowest_distance = distance;
														selected_face_index = i;
														selected_face_bounding_box = curr_face_rect;
													}
												}

											}
											cout << "------------------------------" << endl;
											cout << "Face " << (selected_face_index + 1) << " won!" << endl;
											cout << "Detecting landmarks..." << endl;

											FacialLandmark points[5];
											bool is_there_landmarks = aligner.PointDetectLandmarks(gray_img_data, faces[selected_face_index], points);

											if (is_there_landmarks) {
												cout << "Cropping..." << endl;

												Crop(img, points, img_name);

												cout << "Successful!" << endl;
											}
											else
												cout << "No landmarks detected...";

											/*rectangle(img_copy, selected_face_bounding_box, CV_RGB(0, 0, 255), 4, 8, 0);
											imwrite(LFW_DIR + dst_file + "/" + img_name, img_copy);*/
										}
										else
											cout << "No faces detected!" << endl;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	ifs.close();

	system("pause");
}

void CropWithManuallyExtractedLandmarks(string src_file, string feats_file) {
	cout << "Starting images cropping with manually extracted landmarks..." << endl;

	ifstream src_ifs;
	src_ifs.open(LFW_DIR + src_file, ifstream::in);

	string person_name;

	while (src_ifs >> person_name) {
		string img_name;
		src_ifs >> img_name;

		ifstream feats_ifs;
		feats_ifs.open(LFW_DIR + feats_file, ifstream::in);

		string line;
		FacialLandmark pt5[5];
		int curr_line = 0;

		cout << "========== " << img_name << " ==========" << endl;
		cout << "Finding landmarks..." << endl;

		while (!feats_ifs.eof()) {
			streampos old_pos = feats_ifs.tellg();
			getline(feats_ifs, line);
			curr_line++;
			if (line.find(img_name, 0) != string::npos) {
				feats_ifs.clear();
				feats_ifs.seekg(old_pos);

				string value;
				feats_ifs >> value >> value;

				for (int i = 0; i < 5; ++i)
					feats_ifs >> pt5[i].x >> pt5[i].y;

				cout << "Finding source image..." << endl;

				Mat img = imread(LFW_DIR + IMGS_SRC_DIR + person_name + "/" + img_name, 1);

				if (img.empty())
					cout << "Image not found!" << endl;
				else {
					cout << "Cropping..." << endl;

					Crop(img, pt5, img_name);

					cout << "Image cropped!" << endl;
				}

				break;
			}
		}

		cout << "==============================" << endl;

		feats_ifs.close();
	}

	src_ifs.close();

	cout << "Successful!" << endl;
	system("pause");
}

void SaveFeats(string lfw_list_file, string feats_file) {
	ifstream ifs;
	ifs.open(LFW_DIR + lfw_list_file, ifstream::in);

	filebuf fb;
	fb.open(LFW_DIR + feats_file, ios_base::app);
	ostream os(&fb);

	string person_name;
	string img_file;

	while (ifs >> person_name) {
		ifs >> img_file;
		string data = ExtractFeats(person_name, img_file);
		os << data << endl;
	}
}

void SetNoCroppedImgsList(string dst_file) {
	DIR *src_dir;
	struct dirent *src_ent;

	string src_imgs_dir = LFW_DIR + IMGS_SRC_DIR;
	const char *src_lfw_dir = (src_imgs_dir).c_str();

	string cropped_imgs_dir = LFW_DIR + CROP_DIR;
	const char *cropped_lfw_dir = (cropped_imgs_dir).c_str();

	string no_cropped_imgs_dir = LFW_DIR + dst_file;

	filebuf fb;
	fb.open(no_cropped_imgs_dir, ios_base::app);
	ostream os(&fb);

	if ((src_dir = opendir(src_lfw_dir)) != NULL) {
		while ((src_ent = readdir(src_dir)) != NULL) {
			string directory_name(src_ent->d_name);

			cout << "========== " << directory_name << " ==========" << endl;
			cout << "Searching no cropped images..." << endl;

			if (directory_name.compare(".") != 0 && directory_name.compare("..") != 0) {
				string src_img_path = src_imgs_dir + directory_name;
				const char *src_img_dir = (src_img_path).c_str();

				DIR *src_sub_dir;
				struct dirent *src_sub_ent;

				if ((src_sub_dir = opendir(src_img_dir)) != NULL) {
					while ((src_sub_ent = readdir(src_sub_dir)) != NULL) {
						string img_name(src_sub_ent->d_name);

						if (img_name.compare(".") != 0 && img_name.compare("..") != 0) {
							DIR *crop_dir;
							struct dirent *crop_ent;
							bool found = false;

							if ((crop_dir = opendir(cropped_lfw_dir)) != NULL) {
								while ((crop_ent = readdir(crop_dir)) != NULL) {
									string cropped_img_name(crop_ent->d_name);
									if (cropped_img_name.compare(".") != 0 && cropped_img_name.compare("..") != 0) {
										if (cropped_img_name.compare(img_name) == 0)
										{
											found = true;
											break;
										}
									}
								}
							}
							else {
								cout << "Cropped images directory reading error has ocurred!" << endl;
								closedir(crop_dir);
								break;
							}

							if (!found) {
								os << directory_name << " " << img_name << endl;
								cout << "Image found!" << endl;
							}
							else
								cout << "All " << directory_name << "'s images are cropped!" << endl;

							closedir(crop_dir);
						}
					}
				}
				else {
					cout << "Subdirectory reading error has ocurred!" << endl;
					closedir(src_sub_dir);
					break;
				}

				closedir(src_sub_dir);
			}
		}
	}
	else {
		cout << "Main directory reading error has ocurred!" << endl;
	}

	closedir(src_dir);
	fb.close();

	cout << "Successful" << endl;
	system("pause");
}

void TestFaceDetection(string src_file, int32_t min_size, float thresh, float factor, int32_t setp_x, int32_t setp_y) {
	cout << "Starting face detection..." << endl;

	detector.SetMinFaceSize(min_size); // >= 20
	detector.SetScoreThresh(thresh); // default is 2.0 (could be 0.95, 2.8 or 4.5)
	detector.SetImagePyramidScaleFactor(factor); // 0.7, 0.8 or 1.25
	detector.SetWindowStep(setp_x, setp_y); // 2 or 4

	Mat img;
	img = imread(LFW_DIR + NO_DETECTED_DIR + src_file, 1);

	if (!img.empty()) {
		Mat gray_img;

		if (img.channels() != 1)
			cvtColor(img, gray_img, COLOR_BGR2GRAY);
		else
			gray_img = img;

		seeta::ImageData gray_img_data;
		gray_img_data.data = gray_img.data;
		gray_img_data.width = gray_img.cols;
		gray_img_data.height = gray_img.rows;
		gray_img_data.num_channels = 1;

		vector<FaceInfo> faces = detector.Detect(gray_img_data);

		if (faces.size() == 0)
			cout << "No faces detected..." << endl;
		else {
			cout << "Number of faces detected: " << faces.size() << endl;

			int32_t num_face = static_cast<int32_t>(faces.size());

			int r = 0;
			int g = 0;
			int b = 0;

			Mat img_copy = img.clone();

			Point img_center = Point((img_copy.cols) / 2, (img_copy.rows) / 2);
			cout << "Image center: (" << img_center.x << ", " << img_center.y << ")" << endl;

			cv::circle(img_copy, img_center, 2, CV_RGB(0, 0, 0));

			for (int32_t i = 0; i < num_face; i++) {
				cout << "Face " << (i + 1) << endl;

				cv::Rect face_rect;

				face_rect.x = faces[i].bbox.x;
				face_rect.y = faces[i].bbox.y;
				face_rect.width = faces[i].bbox.width;
				face_rect.height = faces[i].bbox.height;

				if (i == 0)
				{
					r = 255;
					g = 0;
					b = 0;
					cout << "Color: red" << endl;
				}
				else if (i == 1) {
					r = 0;
					g = 255;
					b = 0;
					cout << "Color: green" << endl;
				}
				else {
					r = 0;
					g = 0;
					b = 255;
					cout << "Color: blue" << endl;
				}

				Point face_center = Point(face_rect.x + face_rect.width / 2,
					face_rect.y + face_rect.height / 2);

				cv::circle(img_copy, face_center, 2, CV_RGB(r, g, b), 4, 8, 0);

				double distance = cv::norm(img_center - face_center);
				//double distance = CalcEuclideanDistance(img_center, face_center);

				cout << "Distance: " << distance << endl;
				cout << "Size: " << faces[i].bbox.width *  faces[i].bbox.height << endl;
				cout << "Score: " << faces[i].score << endl << endl;

				cv::rectangle(img_copy, face_rect, CV_RGB(r, g, b), 4, 8, 0);				
			}
			cv::imshow("Image", img_copy);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}			
	}

	cout << "Successful!" << endl;
	system("pause");
}

void TestFaceDetectionWithAllParams(string src_file, string dst_file) {
	int32_t min_size = 40;
	int32_t max_size = 80;
	float thresholds[4] = { 0.95f, 2.0f, 2.8f, 4.5f };
	float factors[3] = { 0.7f, 0.8f, 1.25f };
	int32_t steps[2] = { 2, 4 };

	cout << "Starting face detection..." << endl;

	Mat img;
	img = imread(LFW_DIR + NO_DETECTED_DIR + src_file, 1);

	if (!img.empty()) {
		Mat gray_img;

		if (img.channels() != 1)
			cvtColor(img, gray_img, COLOR_BGR2GRAY);
		else
			gray_img = img;

		seeta::ImageData gray_image_data;
		gray_image_data.data = gray_img.data;
		gray_image_data.width = gray_img.cols;
		gray_image_data.height = gray_img.rows;
		gray_image_data.num_channels = 1;

		int setting_count = 0;

		for (int size = min_size; size <= max_size; size+=5) {
			for (int t = 0; t < (sizeof(thresholds) / sizeof(*thresholds)); t++) {
				for (int f = 0; f < (sizeof(factors) / sizeof(*factors)); f++) {
					for (int s = 0; s < (sizeof(steps) / sizeof(*steps)); s++) {
						detector.SetMinFaceSize(size); // >= 20
						detector.SetScoreThresh(thresholds[t]); // default is 2.0 (could be 0.95, 2.8 or 4.5)
						detector.SetImagePyramidScaleFactor(factors[f]); // 0.7, 0.8 or 1.25
						detector.SetWindowStep(steps[s], steps[s]); // 2 or 4

						cout << "==============================" << endl;
						cout << "Detecting..." << endl;
						
						vector<FaceInfo> faces = detector.Detect(gray_image_data);

						if (faces.size() > 0) {
							setting_count++;
							cout << "Number of faces detected: " << faces.size() << endl;
							cout << "Setting " << setting_count << ":" << endl;
							cout << "   Size: " << size << endl;
							cout << "   Threshold: " << thresholds[t] << endl;
							cout << "   Factor: " << factors[f] << endl;
							cout << "   Step: " << steps[s] << "x" << steps[s] << "\n" << endl;

							cv::Rect face_rect;
							int32_t num_face = static_cast<int32_t>(faces.size());

							Mat copy = img.clone();

							for (int32_t i = 0; i < num_face; i++) {
								face_rect.x = faces[i].bbox.x;
								face_rect.y = faces[i].bbox.y;
								face_rect.width = faces[i].bbox.width;
								face_rect.height = faces[i].bbox.height;

								cv::rectangle(copy, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
							}
							string window_name = "Setting " + to_string(setting_count);

							imwrite(LFW_DIR + dst_file + window_name + ".jpg", copy);
						}
						else {
							cout << "No faces detected..." << endl;
							cout << "Setting " << setting_count << ":" << endl;
							cout << "   Size: " << size << endl;
							cout << "   Threshold: " << thresholds[t] << endl;
							cout << "   Factor: " << factors[f] << endl;
							cout << "   Step: " << steps[s] << "x" << steps[s] << "\n" << endl;
						}
						cout << "==============================" << endl;
					}
				}
			}
			
		}
	}

	cout << "Successful!" << endl;
	system("pause");
}

double CalcEuclideanDistance(Point p1, Point p2) {
	return sqrtf((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2);
}

double CalcProcessingMeanTime(string pairs_file) {
	cout << "***** Processing mean time calculation *****" << endl << endl;

	ifstream ifs;
	ifs.open(LFW_DIR + pairs_file, ifstream::in);

	int k = 0;
	int n = 0;

	ifs >> k;
	ifs >> n;

	int offset = 0;
	bool flag = false;

	double accu = 0;
	int count = 0;

	double secs = 0;
	double ms = 0;
	long ticks = 0;
	long start, end = 0;

	while (!ifs.eof()) {
		string person_name;
		string person_name_2;
		string img_file_1;
		string img_file_2;
		float *img_feats_1;
		float *img_feats_2;
		int n1;
		int n2;

		if (offset == 300) {
			flag = !flag;
			offset = 0;
		}

		if (!flag) {
			ifs >> person_name;
			ifs >> n1;
			ifs >> n2;

			cout << "========== Extracting " << person_name << "'s features... ==========" << endl;

			img_file_1 = SetImgFileName(person_name, n1);

			start = cv::getTickCount();
			img_feats_1 = ExtractFeats(img_file_1);
			end = cv::getTickCount();
			ticks = (end - start);
			secs = ticks / cv::getTickFrequency();
			ms = secs * 1000;
			accu += ms;
			count++;		

			cout << "time 1: " << ms << " ms (" << secs << " s)" <<endl;

			img_file_2 = SetImgFileName(person_name, n2);

			start = cv::getTickCount();
			img_feats_2 = ExtractFeats(img_file_2);
			end = cv::getTickCount();
			ticks = (end - start);
			secs = ticks / cv::getTickFrequency();
			ms = secs * 1000;
			accu += ms;
			count++;
			
			cout << "time 2: " << ms << " ms (" << secs << " s)" << endl;
		}
		else {
			ifs >> person_name;
			ifs >> n1;
			ifs >> person_name_2;
			ifs >> n2;

			cout << "========== Extracting " << person_name << " and " << person_name_2 << "'s features... ==========" << endl;

			img_file_1 = SetImgFileName(person_name, n1);

			start = cv::getTickCount();
			img_feats_1 = ExtractFeats(img_file_1);
			end = cv::getTickCount();
			ticks = (end - start);
			secs = ticks / cv::getTickFrequency();
			ms = secs * 1000;
			accu += ms;
			count++;
			
			cout << "time 1: " << ms << " ms (" << secs << " s)" << endl;

			img_file_2 = SetImgFileName(person_name_2, n2);

			start = cv::getTickCount();
			img_feats_2 = ExtractFeats(img_file_2);
			end = cv::getTickCount();
			ticks = (end - start);
			secs = ticks / cv::getTickFrequency();
			ms = secs * 1000;
			accu += ms;
			count++;

			cout << "time 2: " << ms << " ms (" << secs << " s)" << endl;
		}

		start = cv::getTickCount();
		float sim = recognizer.CalcSimilarity(img_feats_1, img_feats_2);
		end = cv::getTickCount();
		ticks = (end - start);
		secs = ticks / cv::getTickFrequency();
		ms = secs * 1000;
		//accu += end;

		cout << "sim calc time: " << ms << " ms" << endl;
		
		offset++;
	}

	ifs.close();

	double mean_time = accu / count;

	cout << "\nmean time: " << mean_time << " ms" << endl;
	system("pause");

	return mean_time;
}

float* ExtractFeats(string img_file) {
	Mat cropped_img = imread(LFW_DIR + CROP_DIR + img_file, 1);

	if (!cropped_img.empty()) {
		ImageData cropped_img_data;
		cropped_img_data.width = cropped_img.rows;
		cropped_img_data.height = cropped_img.cols;
		cropped_img_data.num_channels = cropped_img.channels();
		cropped_img_data.data = cropped_img.data;

		float* feats = new float[FEAT_SIZE];

		recognizer.ExtractFeature(cropped_img_data, feats);

		return feats;
	}
	return NULL;
}

float* GetFeats(string img_file_name, string feats_file_name) {
	float* feats = new float[FEAT_SIZE];

	ifstream ifs;
	ifs.open(LFW_DIR + feats_file_name, ifstream::in);
	
	string line;
	int curr_line = 0;
	int index = 0;
	
	while (getline(ifs, line)) {
		curr_line++;
		if (line.find(img_file_name, 0) != string::npos) {
			string value;
			for (int i = 0; i < 20; i++) {
				ifs >> value;
				if (i > 12) {
					float x = strtof((value).c_str(), 0);
					float* data_head = &x;
					memcpy(feats, data_head, index * sizeof(float));
					/*feats[index] = strtof((value).c_str(), 0);
					feats[index] = i;*/
					index++;
				}
			}
			break;
		}
	}
	ifs.close();
	return feats;
}

ImageData ConvertMatImageToGrayImageData(Mat img) {
	ImageData gray_img_data;

	if (!img.empty()) {
		Mat gray_img;

		if (img.channels() != 1)
			cvtColor(img, gray_img, COLOR_BGR2GRAY);
		else
			gray_img = img;

		gray_img_data.data = gray_img.data;
		gray_img_data.width = gray_img.cols;
		gray_img_data.height = gray_img.rows;
		gray_img_data.num_channels = 1;
	}

	return gray_img_data;
}

string ExtractFeats(string person_name, string img_file) {
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	Mat img = imread(LFW_DIR + IMGS_SRC_DIR + person_name + "/" + img_file, IMREAD_UNCHANGED);
	string data = "";

	if (!img.empty()) {
		Mat gray_img;

		if (img.channels() != 1)
			cvtColor(img, gray_img, COLOR_BGR2GRAY);
		else
			gray_img = img;

		seeta::ImageData gray_img_data;
		gray_img_data.data = gray_img.data;
		gray_img_data.width = gray_img.cols;
		gray_img_data.height = gray_img.rows;
		gray_img_data.num_channels = 1;

		cout << "Detecting face..." << endl;

		vector<FaceInfo> faces = detector.Detect(gray_img_data);

		int32_t probe_face_num;
		if (faces.size() == 0)
		{
			probe_face_num = static_cast<int32_t>(faces.size());
			cout << "No faces detected..." << endl;
		}
		else {
			cout << "Detecting landmarks..." << endl;

			FacialLandmark points[5];
			bool flag = aligner.PointDetectLandmarks(gray_img_data, faces[0], points);

			if (!flag) {
				cout << "No landmarks detected..." << endl;
			}
			else {
				seeta::ImageData img_data;
				img_data.data = img.data;
				img_data.width = img.cols;
				img_data.height = img.rows;
				img_data.num_channels = img.channels();

				Mat dst_img(recognizer.crop_height(), recognizer.crop_width(), CV_8UC(recognizer.crop_channels()));

				seeta::ImageData cropped_img;
				cropped_img.data = dst_img.data;
				cropped_img.width = dst_img.cols;
				cropped_img.height = dst_img.rows;
				cropped_img.num_channels = dst_img.channels();

				cout << "Cropping image..." << endl;

				recognizer.CropFace(img_data, points, cropped_img);

				string cropped_img_path = LFW_DIR + CROP_DIR + img_file;
				imwrite(cropped_img_path, dst_img);

				float* feats;

				if (recognizer.feature_size() != FEAT_SIZE)
					cout << "Feature size is not equal to 2048..." << endl;

				cout << "Extracting features with crop..." << endl;

				feats = new float[FEAT_SIZE];
				recognizer.ExtractFeatureWithCrop(cropped_img, points, feats);

				for (int i = 0; i < sizeof(points) / sizeof(*points); i++)
					data += " " + to_string(points[i].x) + " " + to_string(points[i].y);
				for (int i = 0; i < sizeof(feats); i++)
					data += " " + to_string(feats[i]);

				cout << "Successful!\n" << endl;

				return data;
			}
		}

		data = "";
		for (int i = 0; i < 10; i++)
			data += " 0.000000 ";
		for (int i = 0; i < 8; i++)
			data += " 0.000000";

		cout << "Process completed with errors! " << endl;

		return "";
	}
}

string SetImgFileName(string img_file_name, int number) {
	string complement = "_";
	string image_ext = ".jpg";
	string src_name;
	string zeros = "";
	if (number < 10)
		zeros = "000";
	else if (number >= 10 && number < 100)
		zeros = "00";
	else if (number >= 100 && number < 1000)
		zeros = "0";
	return img_file_name + complement + zeros + to_string(number) + image_ext;
}

wstring s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

#pragma region Commented
//int read_lfw() {
//	long ts = 0;
//	long te = 0;
//	double secs = 0;
//
//	ts = cv::getTickCount();
//
//	DIR *dir;
//	struct dirent *ent;
//
//	// directory of directories of all images on lfw database
//	std::string images_dir = LFW_DIR + "images/";
//	const char *lfw_dir = (images_dir).c_str();
//	// directory of data
//	std::string lfw_file_list_dir = LFW_DIR + "lfw_file_list.txt";
//
//	std::string image_name;
//
//	// reader
//	std::ifstream ifs;
//	ifs.open(lfw_file_list_dir, std::ifstream::in);
//	// start read in the second line
//
//	// writter
//	std::filebuf fb;
//	fb.open(lfw_file_list_dir, std::ios_base::app);
//	std::ostream os(&fb);
//
//	if ((dir = opendir(lfw_dir)) != NULL) {
//		while ((ent = readdir(dir)) != NULL) {
//			// read all subdirectories
//			std::string directory_name(ent->d_name);
//			if (directory_name.compare(".") != 0 && directory_name.compare("..") != 0) {
//				ifs >> image_name;
//				//if (image_name.size() == 0) {
//				// this is for subdirectories
//				DIR *sub_dir;
//				struct dirent *sub_ent;
//				// subdirectory of one specific person
//				std::string images_path = images_dir + directory_name + "/";
//				if ((sub_dir = opendir(images_path.c_str())) != NULL) {
//					// read all image files into subdirectory
//					while ((sub_ent = readdir(sub_dir)) != NULL) {
//						// get current file name
//						std::string file_name(sub_ent->d_name);
//						if (file_name.compare(".") != 0 && file_name.compare("..") != 0) {
//							// build the entire path until the current image
//							std::string current_image_path = images_path + file_name;
//							// get image landmarks and feats
//							string data = test(current_image_path, file_name);
//							// write on directory of data
//							os << directory_name << " " << file_name << " " << data << endl;
//						}
//					}
//					closedir(sub_dir);
//				}
//				else {
//					perror("");
//					return EXIT_FAILURE;
//				}
//			}
//		}
//	}
//	else {
//		perror("");
//		return EXIT_FAILURE;
//	}
//
//	closedir(dir);
//	ifs.close();
//	fb.close();
//
//	te = cv::getTickCount();
//	secs = (te - ts) / cv::getTickFrequency();
//
//	cout << "\nTotal time: " << secs << "s" << endl;
//	system("pause");
//}
#pragma endregion