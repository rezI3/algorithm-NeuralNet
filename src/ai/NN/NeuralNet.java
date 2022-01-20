package ai.NN;

import java.util.Random;

public class NeuralNet {

	final int N_INPUT;  // 入力層の数
	final int N_HIDDEN; // 中間層の数
	final int N_OUTPUT;  // 出力層の数

	double w1[][];	//入力層>隠れ層の重み
	double w2[][];	//隠れ層>出力層の重み
	double b1[];	//定数1を入力とした入力層>隠れ層の重み 数式的には閾値θとした時の-θと等しい
	double b2[];	//定数1を入力とした隠れ層>出力層の重み 数式的には閾値θとした時の-θと等しい

	double input[];  // 入力層
	double hidden[]; // 中間層
	double output[]; // 出力層

	/* ③ */
	double alpha = 0.1;	//学習率

						//36,36,4
	public NeuralNet( int nInput, int nHidden, int nOutput ) {
		//入力層、中間層、出力層の数を引数で決定
		N_INPUT = nInput;
		N_HIDDEN = nHidden;
		N_OUTPUT = nOutput;

		//それぞれの層を要素数が引数の配列として宣言
		input  = new double[N_INPUT];  // 入力層
		hidden = new double[N_HIDDEN]; // 中間層
		output = new double[N_OUTPUT]; // 出力層

		// 重みを-0.1~0.1で初期化
		Random rnd = new Random();

		//入力層と中間層の間の重みを乱数で初期化

		/* ① */
		w1 = new double[N_INPUT][N_HIDDEN];
		for(int i=0; i<N_INPUT; i++){
			for(int j=0; j<N_HIDDEN; j++){
				//			(0.0~1.0の乱数　* 2 - 1 ) * 0.1
				//			(-1.0~1.0の乱数)　* 0.1 => -0.1~0.1の乱数
				w1[i][j] = (rnd.nextDouble()*2.0 - 1.0) * 0.1;
			}
		}

		// 1を入力した時の入力層と中間層の間の重み
		b1 = new double[N_HIDDEN];

		//中間層と出力層の間の重みを乱数で初期化
		w2 = new double[N_HIDDEN][N_OUTPUT];
		for(int i=0; i<N_HIDDEN; i++){
			for(int j=0; j<N_OUTPUT; j++){
				w2[i][j] = (rnd.nextDouble()*2.0 - 1.0) * 0.1;
			}
		}
		// 1を入力した時の中間層と出力層の間の重み
		b2 = new double[N_OUTPUT];

	}

	// NNに入力し、出力を計算する
	// 引数が入力[36]　戻り値が出力[4]
	public double[] compute(double x[]){

		// 入力層の入力
		// 引数を入力層への入力として受け取る
		for(int i=0; i<N_INPUT; i++){
			input[i] = x[i];
		}

		// 中間層の計算
		for(int i=0; i<N_HIDDEN; i++){

			//まず中間層の中身を0にする
			hidden[i] = 0.0;
			for(int j=0; j<N_INPUT; j++){

				//i番目の中間層の中身を、重み*入力の値 の総和にする
				/* ④ */
				hidden[i] += w1[j][i] * input[j];
			}

			hidden[i] += b1[i]; //1を入力した時の入力層と中間層の間の重みを足す

			/* ⑤ */
			hidden[i] = sigmoid(hidden[i]);//シグモイド関数に入れ、その出力を入れる
		}

		// 出力層の計算
		for(int i=0; i<N_OUTPUT; i++){

			//まず出力層の中身を0にする
			output[i] = 0.0;
			for(int j=0; j<N_HIDDEN; j++){
				//出力層の中身を 重み*中間層の中身 の総和にする
				/* ⑥ */
				output[i] += w2[j][i] * hidden[j];
			}

			output[i] += b2[i]; // 1を入力した時の中間層と出力層の重みを加える

			/* ⑦ */
			output[i] = sigmoid(output[i]); //シグモイド関数に入力し、その出力を入れる
		}

		return output;
	}

	// シグモイド関数
	public double sigmoid(double i){
		double a = 1.0 / (1.0 + Math.exp(-i));
		return a;
	}

	// 誤差逆伝播法による重みの更新
	public void backPropagation(double teach[]){

		// 誤差
		double[] deltas = new double[N_OUTPUT];

		// 中間層>出力層の重みを更新
		for(int j=0; j<N_OUTPUT; j++){
			// 誤差 = (教師信号-出力信号) * 出力信号 * (1-出力信号)
			//		= (y0 - yt) * y0 * (1-y0) <- 偏微分の結果
			/* ⑧ */
			deltas[j] = (teach[j]-output[j]) * output[j] * (1.0-output[j]);
			for(int i=0; i<N_HIDDEN; i++){
				// 勾配法????
				// w[i][j] -= ε * alpha * deltas[j] * hidden[i] じゃないのかしら
				/* ⑨ */
				w2[i][j] += alpha * deltas[j] * hidden[i];
			}

			// 1が入力された時の重みに、alpha * 誤差のw[i][j]での微分　を足す ??
			// 何をしているのかわからない
			b2[j] += alpha * deltas[j];
		}

		// 入力層>中間層の重みを更新
		for(int i=0; i<N_HIDDEN; i++){

			double sum = 0.0;
			for(int j=0; j<N_OUTPUT; j++){
				// 中間層と出力層の重み * 出力層から中間層に至るまでの偏微分の結果　の総和
				sum += w2[i][j] * deltas[j]; //誤差の逆伝播
			}

			// deltaに今度は　sum * 中間層の値 * (1 - 中間層の値) を入れる
			/* ⑧ */
			double delta = hidden[i] * (1.0-hidden[i]) * sum;

			for(int j=0; j<N_INPUT; j++){
				//勾配法??
				// w[j][i] = w[j][i] - ε * 偏微分の結果
				/* ⑨ */
				w1[j][i] += alpha * delta * input[j];
			}
			b1[i] += alpha * delta;
		}
	}

	// 二乗誤差
	public double calcError(double teach[]){
		// 誤差e を一旦0で初期化
		double e = 0.0;
		for(int i=0; i<teach.length; i++){
			//　各　教師信号と出力信号の誤差の2乗　の総和を求めている
			e += Math.pow(teach[i]-output[i], 2.0);
		}
		//　それの半分　(1/2をかけてる　->　微分した時に2を消せるように)
		e *= 0.5;
		return e;
	}

	// 学習
	public void learn( double[][] knownInputs, double[][] teach ) {

		int step = 0; //試行回数
		while ( true ) {

			double e = 0.0; // 二乗誤差の総和(初期値は0.0)

			// すべての訓練データをニューラルネットワークに入力・計算・誤差伝搬
			for(int i=0; i<knownInputs.length; i++){
				compute(knownInputs[i]);  // データを入力して、出力を出す

				//以下2行は順不同?
				backPropagation(teach[i]); // 誤差逆伝搬法による重みの更新

				/* ⑩ */
				e += calcError(teach[i]); // 二乗誤差をeに格納


			}

			// 100刻みで誤差を表示
			if ( step % 100 == 0) {
				System.out.println("step:" + step + ", loss=" + e);
			}

			// 二乗誤差が十分小さくなったら、終了
			/* 11 */
			if(e < 0.0001){
				break;
			}

			step++;
		}

	}

}





