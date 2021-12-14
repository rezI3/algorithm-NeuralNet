package ai;

/**
 * 文字認識(Character Recognition)
 * @author tkato
 *
 */

// TODO:入力を0-9までにする
// TODO:GUIによる手書き認識を実装
// TODO:ランプ関数
// TODO:ソフトマックス関数でワンホットベクトル
public class CRApp09 {

    static final int MASS_X = 6; // マス目の数（縦）
    static final int MASS_Y = 6; // マス目の数（横）

    public static void main(String[] args) {

        NeuralNet nn = new NeuralNet(36, 36, 4);

        // 訓練データ（入力）
        double knownInputs[][] = {
                { // 0の訓練データ
                        0, 0, 1, 1, 0, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0,
                        0, 0, 1, 1, 0 ,0
                },
                { // 1の訓練データ
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0 ,0
                },
                { // 2の訓練データ
                        0, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0 ,0

                }
        };

        // 教師データ
        double t[][] = {
                {0, 0, 0, 0}, // この組み合わせを0とする
                {0, 0, 0, 1},  // この組み合わせを1とする
                {0, 0, 1, 0}   // この組み合わせを2とする
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
                        0, 0, 1, 1, 0 ,0
                },
                { // 1の入力データ
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0 ,0
                },
                { // 2の入力データ
                        0, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 1, 1, 1, 1 ,0

                }

        };

        // 教師データ
        double expects[][] = {
                {0, 0, 0, 0}, // この組み合わせを0とする
                {0, 0, 0, 1},  // この組み合わせを1とする
                {0, 0, 1, 0}    //この組み合わせを2とする
        };

        for ( int i=0; i<unknownInputs.length; i++ ) {
            double[] output = nn.compute(unknownInputs[i]);
            print(unknownInputs[i], output, expects[i]);
        }
        System.out.println("\n--推論終了--");

    }

    // 画面に入力データと実体値、予測値を表示する
    public static void print(double[] input, double[] output, double[] expect) {
        System.out.println();
        System.out.println("入力データ");
        for (int j = 0; j < MASS_Y; j++) {
            for (int k=0; k<MASS_X; k++) {
                System.out.print((int)input[j*MASS_X+k]);
                System.out.print(" ");
            }
            System.out.println();
        }
        System.out.println("期待する値（文字）：" + value(expect) );
        System.out.println("ニューラルネットワークが予測した値（文字）：" + value(output));
    }

    // 出力データと数字のマッピングを行う
    public static int value(double[] a){
        // ステップ関数（0.5を閾値として0,1に変換）
        //  if     0 <= an < 0.5   then  0
        //  if  0.5 <   an <    1  then  1
        // 小数をintに入れることで、四捨五入させてる？？
        int v1 = (int)(a[0]+0.5);
        int v2 = (int)(a[1]+0.5);
        int v3 = (int)(a[2]+0.5);
        int v4 = (int)(a[3]+0.5);

        // 出力層の値
        return v1 * 8 + v2 * 4 + v3 * 2 + v4 * 1;
    }

}

