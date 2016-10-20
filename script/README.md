## `plot_log_cost.py`
- log costのプロット
- Usage: `python plot_log_cost.py trin.log`

## `translate_all.sh`
- 保存されたモデルに対して翻訳を事項
- Usage: `translate_all.sh ../models/nmt/ nmt tmp 1000 1000`
    - `../model/nmt` : モデルの保存先
    - `nmt` : モデルのprefix (モデルは`model_$PEFIX.iter*.npz`として保存されてると仮定)
    - `tmp` : 出力先
    - `1000` : 最初のモデルの数字
    - `1000` : ステップ
    - 上の例だと，`../models/nmt/model_nmt.iter1000.npz`, `../models/nmt/model_nmt.iter2000.npz`, ... が処理される
- 計算時間短縮のためデフォルトでビーム幅は1

## `eval_result_all.sh`
- METEORスコアを計算
- Usage: `bash eval_result_sll.sh meteor-1.5.jar tmp nmt`
    - `meteor-1.5.jar` : METERORのjarファイル
    - `tmp` : 翻訳が保存されているファイル
    - `nmt` : prefix
- グラフにプロットするには，適当なファイルに結果を保存して `gnuplot -e "set  terminal png; set output 'meteor.png'; plot 'nmt_meteor.txt' with linespoints"`

