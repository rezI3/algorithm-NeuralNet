package ai;

import java.util.Arrays;


//・ 入力01から012に増やした　（3を入れると誤差がうまく減らず動かなくなったため、0から9にはできなかった）
//・ ワンホットベクトルを導入した
//・ ReLU関数を実装した
//・ Softmax関数を実装した
//・ 誤差関数を交差エントロピー誤差にした

public class CRReluSoftmaxApp {

    static final int MASS_X = 6; // マス目の数（縦）
    static final int MASS_Y = 6; // マス目の数（横）

    public static void main(String[] args) {

        NNReluSoftmax nn = new NNReluSoftmax(36, 72, 3);

        // 訓練データ（入力）
        /* ② */
        double knownInputs[][] = {
                { // 0の訓練データ
                        0, 0, 1, 1, 0, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 0, 1, 1, 0, 0
                },
                { // 1の訓練データ
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0
                },
                { // 2の訓練データ
                        0, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0
                }

        };

        // 教師データ
        double t[][] = {
                {1, 0, 0}, // この組み合わせを0とする
                {0, 1, 0}, // この組み合わせを1とする
                {0, 0, 1}, // この組み合わせを2とする
        };

        // 学習
        System.out.println("--学習開始--");
        nn.learn(knownInputs, t);
        System.out.println("--学習終了--");

        System.out.println("\n--推論開始--");
        // ---------------------
        // 推定はここから
        // ---------------------
        double[][] unknownInputs = {
                { // 0の入力データ
                        0, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 0, 1, 1, 0, 0
                },
                { // 1の入力データ
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0
                },
                { // 2の入力データ
                        0, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 0

                }
        };

        // 教師データ
        double expects[][] = {
                {1, 0, 0}, // この組み合わせを0とする
                {0, 1, 0}, // この組み合わせを1とする
                {0, 0, 1}, // この組み合わせを2とする
        };

        for (int i = 0; i < unknownInputs.length; i++) {
            /* 12 */
            double[] output = nn.compute(unknownInputs[i]);
            print(unknownInputs[i], output, expects[i]);
        }
        System.out.println("\n--推論終了--");

    }

    // 画面に入力データと実体値、予測値を表示する
    public static void print(double[] input, double[] output, double[] expect) {
        System.out.println();
        // 入力データの表示
        System.out.println("入力データ");
        for (int j = 0; j < MASS_Y; j++) {
            for (int k = 0; k < MASS_X; k++) {
                if(input[j * MASS_X + k] == 1){
                    System.out.print("\u001b[00;44m  \u001b[00m");
                }else {
                    System.out.print("\u001b[00;40m  \u001b[00m");
                }
            }
            System.out.println();
        }
        // ソフトマックス関数でのそれぞれに対する確率の表示
        int i = 0;
        for (double out : output) {
            System.out.print("[" + i + "] : " + String.format( "%.2f", out * 100) + "% ");
            i++;
        }
        System.out.println();

        // 教師データと推論結果を表示
        System.out.println("期待する値（文字）：" + value(expect));
        System.out.println("ニューラルネットワークが予測した値（文字）：" + value(output));
    }

    // 出力データと数字のマッピングを行う
    public static int value(double[] a) {
        // 出力データの中で最大値であるものを取得
        double max = Arrays.stream(a).max().getAsDouble();

        //  最大値であるものの要素番号を取得 (最大値の要素番号 = Onehotベクトルが示す値)
        int index = 0;
        for (double j : a) {
            if (j == max) {
                break;
            }
            index++;
        }
        return index;
    }

}
