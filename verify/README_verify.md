# meshkit 動作確認キット

meshkit を別のPCや新しい Python 環境に移したときに、作った時と同じように動いているかを確かめるためのものです。
確認用のモデルを meshkit で作り直し、表示し、その結果を同梱の基準データ（比較用画像など）と照らし合わせます。

## 中身

| 場所 | 内容 |
|---|---|
| `verify.py` | 確認スクリプト。すべての検査を行い、結果を `result/` に書き出します |
| `models/` | 確認用モデル8体。それぞれ OBJ（＋MTL、テクスチャPNG）と GLB があります。`calib_card.py` は確認専用モデルのスクリプトです |
| `reference/` | 基準データ。モデルごとの比較用画像（`<モデル名>/<視点>_<表示モード>.png`）、一覧画像（`sheet_<モデル名>.png`）、基準の数値（`reference.json`） |
| `reference/blender/` | 同じ確認用モデルを Blender 5.2 で同じカメラから描いた画像（別の描画ソフトでの見え方の参考） |
| `blender_render.py` | Blender で確認用モデルを描き、輪郭を比べるためのスクリプト（任意） |

確認用モデルは次のとおりです。

- **calib_card**（確認専用の「テストカード」）
  - **座標軸の矢印**：赤＝+X（モデルの左）、緑＝+Y（上）、青＝+Z（前）。軸の入れ替わりや左右の反転は、矢印の色と向きで分かります。
  - **台の上面**：0.5mの市松模様です。前左の角（+X、+Z）は赤いマス、右半分には黄色の「F」があります。テクスチャのずれや裏返しが分かります。
  - **基本形状**：箱、円柱、円錐（回転体）、楕円体、断面ロフト、翼、プロペラ羽根です。
  - **その他**：半透明のドーム、回転中心つきのふた（`$lid`）があります。
- **light_tank**：例のスクリプト `examples/light_tank.py` です。
- **j7w1, kikka, j8m1, xb42, xf15c, xf5u**：`examples/ww2_projects/` の6機です。

## 確認のしかた

numpy と Pillow が入っていれば動きます（Blender は不要です）。meshkit フォルダで次を実行します。

```
python verify/verify.py
```

約40秒で終わります。`--quick` を付けると、calib_card と light_tank だけを確認します（約10秒）。

最後に次のような結果が出ます。

```
結果         : PASS  (完全一致 134 / 許容内 0 / 不一致 0)
```

- **PASS**：不一致が0件です。この環境で meshkit は基準と同じように動いています。
- **FAIL**：不一致の項目名が表示されます。詳しくは `result/report.json` と比較画像を見てください。

戻り値は、PASS のとき0、FAIL のとき1です。

### 目で見て比べる

`result/compare/<モデル名>.png` は、検査した表示1つにつき1行です。左から順に次の3つが並びます。

1. 基準の画像（`reference/` と同じもの）
2. この環境で描いた画像
3. 差分（違う画素が赤、ごくわずかな違いは橙）

各行の見出しに、判定（MATCH / OK / FAIL）が出ます。

描いた画像そのものは `result/render/` にあります。

確認用モデルを Blender など別のソフトで開くこともできます（`models/*.glb` または `models/*.obj`。どちらも Blender の既定の読み込み設定で正しい向きになります）。そのうえで `reference/sheet_<モデル名>.png` や `reference/blender/` と見比べることもできます。

## 何を確かめているか

| 検査 | 内容 | 判定 |
|---|---|---|
| build | モデルのスクリプトを実行し直して、書き出した OBJ を `models/` の OBJ と比べます。テクスチャも画素単位で比べます | 一致すれば MATCH。座標・UVの違いが1e-5以下で面の並びが同じなら OK |
| load | `models/` の OBJ と GLB を読み込み、三角形数、パーツ、寸法、回転中心、参照点、テクスチャの大きさを比べます。メッシュ検査（穴・裏返りなど）も行います | 記録と同じなら MATCH |
| render | 読み込んだモデルを、決まったカメラと表示モード（texture / parts / solid / tags / orientation / xray、ワイヤーフレーム、参照点の印、パーツを回した姿勢）で描き、基準画像と比べます。GLB から読んだモデルも描きます | 画素がすべて同じなら MATCH。違う画素（どれかの色が8より大きく違う）が0.2%以下で、輪郭の重なり（IoU）が0.995以上なら OK |
| cli | `python -m meshkit info / check / convert` を別のプロセスとして実行し、結果を確かめます | 同上 |
| blender | （任意）Blender の画像の輪郭が meshkit の画像と重なる割合（IoU）を見ます | 0.90以上で OK |

判定の値は `verify.py` の `TOL` にまとめてあります。

### 別の環境で「許容内（OK）」になる場合

numpy や Pillow のバージョンが違うと、計算の丸めの違いで、輪郭の数画素だけが変わることがあります。このときは OK になります。
基準の環境（`reference.json` の `environment`）では、すべて MATCH です。

参考までに、これまでに次の2つの環境で、モデル（OBJ）とテクスチャの画素が完全に一致することを確かめています。PNG ファイルのバイト列は Pillow の圧縮の違いで変わりますが、比較は画素で行っているので、影響はありません。

- Windows（Python 3.12、numpy 2.5.3、Pillow 10.2.0）
- Linux（Python 3.13、numpy 2.5.3、Pillow 12.3.0）

このキットそのもの（`verify.py`）は、基準を作った環境と、numpy 2.1.3・Pillow 10.4.0 に入れ替えた環境で実行しました。どちらも134項目すべてが完全一致（MATCH）でした。

## Blender での確認（任意）

Blender は、確認のためだけに使います。モデリングには使いません。

```
blender -b -P verify/blender_render.py -- --out verify/result/blender
python verify/verify.py --blender verify/result/blender
```

1. 1つ目のコマンドで、`models/*.glb` を Blender の glTF 読み込みで開き、基準画像と同じカメラで Workbench 描画します（背景は透明）。
2. 2つ目のコマンドで、その輪郭を meshkit の画像と比べます。

`result/compare/blender_<モデル名>.png` には、次の3つが並びます。

1. meshkit の画像
2. Blender の画像
3. 輪郭の重なり（赤：meshkit だけ、青：Blender だけ）

同梱の `reference/blender/` は、作者の環境（Blender 5.2.2）でこの手順を実行した結果です。

- `<モデル名>_<視点>.png`：Blender で描いた画像です。
- `compare_<モデル名>.png`：比較画像です。
- `blender_report.json`：輪郭の重なり（IoU）です。40枚すべてが0.94〜1.00でした。

Blender がない環境でも、これらの画像を見れば、確認用モデルが別のソフトでも同じ形・向き・大きさで読み込まれることが分かります。

描き方の違いがあるため、完全には一致しません。細い部品の太さや、半透明の表し方（meshkit は mod と同じディザ）が違います。輪郭の位置・向き・大きさがそろっていれば、読み書きと座標変換は正しく動いています。

## 基準データを作り直す（作者向け）

meshkit 本体や確認用モデルを意図して変えたときだけ、次を実行します。

```
python verify/verify.py --make-reference --yes
```

`models/` と `reference/` を、今のコードで作り直します。それまでの基準は上書きされます。
