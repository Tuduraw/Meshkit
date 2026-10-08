# meshkit

Blenderを使わずに、AIがスクリプトでローポリモデルを作るための道具です。
形を作る、検査する、プレビュー画像を出す、OBJ / glTF に書き出す、の4つをPythonだけで行います。
必要なのは numpy と Pillow だけなので、Windows・Linux・Mac・クラウドのどこでも同じように動きます。

ww2gen（vehicleaddonww2の生成スクリプト）で使ってきた汎用部分（閉じたメッシュの形状、検査、投影テクスチャ、modと同じ描き方のプレビュー）を独立させ、Blenderに頼っていた書き出しと表示を置き換えたものです。

## できること

| 機能 | 内容 |
|---|---|
| 形状 | 箱、円柱、回転体、断面ロフト、押し出し、楕円体、翼・尾翼・プロペラ羽根。どれも閉じた外向きのシェルを返す |
| 検査 | 穴、巻き順、裏返り、つぶれた面、平らでない面、凹の面、浮き部品（接触判定）、外形寸法、三角形数 |
| テクスチャ | 上・下・側面・正面の投影アトラスと単色の色見本。塗装は3D座標の関数で書く |
| プレビュー | ソフトウェア描画（背面カリング、面単位の陰影、ガラスのディザ）。透視の斜め視点、正投影の六面図、表示モード6種、ワイヤーフレーム、参照点の印 |
| 入出力 | OBJ（Blender 4.5の書き出しと同じ並び）、glb/glTF（読み書き。パーツごとのノード、原点＝回転中心、テクスチャ埋め込み）、Blender座標（Z上）との変換 |
| 現代兵器用キット（0.3.0） | 直線の台形翼（公表値から翼弦を算出）、ノズルを内包する双発機の後部胴体、コックピットのくぼみ、引き込み脚の自動収納と脚収納部の膨らみ、戦車の車体側面形（傾斜角から）・足回り・履帯経路・張り出し付き砲塔・搭載モジュール、現代艦の傾斜壁の上部構造と艤装、墨入れなどテクスチャの書き込み、めり込み・公表寸法・標識位置の検査。手引きは `meshkit/AI_GUIDE.md` の6章、例は `examples/modern/` |
| ロボット用キット（0.4.0、0.4.1で追加） | 4つのデザイン言語の断面（丸い五角形、管と格子桁、稜線のある六角形、鋳造の超楕円）、斜材付きの桁と支柱で浮かせた別体の装甲板（0.4.1）、装甲板・スカート、関節の回転（変形の姿勢づくり）、充填率・正面幅/側面奥行き・脚の寸法の数値化。手引きは `meshkit/AI_GUIDE.md` の7章、例は `examples/mech/concept_legs.py` |
| 窓口 | コマンドライン（`python -m meshkit ...`）と、MCPサーバー（`python -m meshkit.mcp_server`） |

表示モード：`texture`（テクスチャ）、`parts`（パーツごとに色分け）、`solid`（灰色の粘土）、`tags`（面のタグごとに色分け）、`orientation`（裏面を赤で表示。穴や裏返りが見える）、`xray`（透けて見える）。

## 準備

```
pip install numpy Pillow            # 必須
pip install opencv-python-headless  # 図面との輪郭比較（silhouette）を使う場合のみ
```

インストールせずに、このフォルダを `PYTHONPATH` に入れて使えます。`pip install -e .` でもかまいません（`meshkit` と `meshkit-mcp` のコマンドが入ります）。

## 使い方

### 動作確認（別のPC・環境へ移したとき）

```
python verify/verify.py
```

確認用モデル8体を作り直して描き、同梱の比較用画像・基準データと照合します（約40秒、PASS / FAIL を表示）。
比較画像は `verify/result/compare/` に出ます。手順と判定の基準は `verify/README_verify.md` を見てください。

### スクリプトを書いて実行する

```python
# tank.py
import meshkit as mk

def build():
    m = mk.Model("tank")
    m.add("hull", mk.box(0, 0.7, 0, 2.3, 0.8, 4.6, tag="hull"))
    m.add("$turret", mk.cylinder((0, 1.1, -0.2), (0, 1.6, -0.2), 0.8, 0.65, n=12, tag="turret"))
    m.pivots["$turret"] = (0, 1.1, -0.2)
    return m
```

```
python -m meshkit run tank.py -o out --mode parts --wire
```

検査結果（JSON）が表示され、`out/` に `tank.obj`、`tank.glb`、`tank_preview.png` ができます。
もっと細かい例は `examples/light_tank.py`（パーツ分け、回転中心、テクスチャと塗装、参照点）です。

### 既存のモデルを見る・調べる

```
python -m meshkit render model.obj --texture model.png --views iso,iso_rear,left,top
python -m meshkit render blender_export.obj --axes blender --mode orientation
python -m meshkit check model.glb
python -m meshkit info model.obj
python -m meshkit convert model.obj model.glb --texture model.png
python -m meshkit diff old.obj new.obj
```

### MCPサーバーとして使う（Claude Desktop / Claude Code）

```json
{
  "mcpServers": {
    "meshkit": {
      "command": "python",
      "args": ["-m", "meshkit.mcp_server", "--workspace", "C:/work/meshkit_work"],
      "env": {"PYTHONPATH": "C:/path/to/meshkit"}
    }
  }
}
```

ツール：`guide`（作業ガイド）、`run_script`（コードを保存して実行し、検査結果とプレビュー画像を返す）、`render`、`check`、`info`、`convert`、`list_files`。
スクリプトは別プロセスで実行するので、スクリプトの誤りや無限ループでサーバーが止まることはありません（既定のタイムアウト300秒）。

クラウド環境など、MCPを登録できない場所では、コマンドラインを実行してプレビュー画像を読むだけで同じ作業ができます。

### AI向けの手引き

`meshkit/AI_GUIDE.md` に、座標系、API、作業の流れ、守るべき規則（閉じたメッシュ、食い込ませてつなぐ、凸の面、背面カリング前提）をまとめています。MCPの `guide` ツールも同じ内容を返します。

## ww2gen での使い方

`vehicleaddonww2/tools/ww2gen/build.py` は、Blenderの `bpy` モジュールがない環境では自動で meshkit を使います（`lib/mkio.py`）。

- OBJ は Blender の書き出しと同じ並びで書き出します。
- `.blend` の代わりに、同じ場所へ `.glb` を書き出します。パーツごとのノードで、原点は回転中心、テクスチャ埋め込み、座席・銃口・カメラは空のノードです。Blenderで読み込めます。
- 環境変数 `WW2_BACKEND=meshkit` または `blender` で、どちらを使うか指定できます。
- meshkit の場所は、`modeling/meshkit`（vehicleaddonww2 と同じ階層）か、環境変数 `MESHKIT_PATH` です。

## 検証結果（2026-10-04）

`tests/compare_blender.py` で、ww2gen の4機種をBlenderなしで作り、公開済みパック（アップデート21、Blender 4.5.3で書き出し）のOBJと比べました。

| 機種 | 頂点 | UV | 法線 | 面の並び |
|---|---|---|---|---|
| a6m2（零戦二一型） | 919個すべて一致（差 ≦ 0.000001） | 一致 | 一致（差 ≦ 0.0001） | 完全一致 |
| chiha（九七式中戦車） | 1,003個すべて一致 | 一致 | 一致 | 完全一致 |
| fubuki（吹雪） | 2,292個すべて一致（差 ≦ 0.000003） | 一致 | 4個だけ重複判定が異なる（Blenderの単精度の丸め） | 番号のずれのみ |
| yamato（大和） | 5,669個すべて一致（差 ≦ 0.000007） | 4隅だけ違う（テクスチャ配置の判定差） | 一致（差 ≦ 0.001。重複判定の違い） | 番号のずれのみ |

さらに、a6m2 は build.py をBlenderなしで最後まで実行し、テクスチャPNGとJSONが公開済みのものとバイト単位で一致することを確かめました。

表示の確認として、meshkit が書き出した a6m2.glb をBlender 5.2に読み込み（17パーツ・1,642三角形、OBJも同数）、meshkit の `iso` と同じカメラでWorkbench描画して比べました。輪郭の一致率（IoU）は0.96です。
違いはキャノピーのガラスだけで、meshkit はmodと同じディザで半透明を表し、Blender はglTFのアルファしきい値（0.5）で不透明に表示します。

`tests/selftest.py` は、例のモデルの作成・検査、OBJ / GLB / Blender座標の往復、全視点・全表示モードの描画、欠陥（穴・裏返り・浮き部品）の検出を確かめます。

## Blenderとの違い・制限

- ブーリアン、ベベル、細分化、スムーズシェーディングはありません。閉じたシェルを食い込ませて組み立てる作り方を前提にしています。
- 描画は確認用です。EEVEE/Cyclesのような見栄えではなく、modでの見え方に合わせています。
- 対話的な手作業の編集はできません。人が手で直したいときは、書き出したOBJ / GLBをBlenderで開いてください。
- アニメーションやリグは扱いません（可動部の回転中心と、姿勢を変えたプレビューのみ）。
- `.blend` は読み書きしません。

## 構成

```
meshkit/
  meshkit/
    geom.py        形状（Shell / Part / Model、プリミティブ）
    check.py       メッシュ検査と report()
    attach.py      接触判定（浮き部品）
    texture.py     投影アトラスと texture_model()
    render.py      ソフトウェア描画
    view.py        視点・表示モード・ワイヤーフレーム・まとめ画像
    silhouette.py  図面と重ねる輪郭（opencv）
    planform.py    直線の台形翼・尾翼・垂直尾翼（0.3.0）
    fuselage.py    胴体・ナセル・キャノピー・コックピットのくぼみ・ノズル（0.3.0）
    gear.py        引き込み脚の収納探索と脚収納部の膨らみ（0.3.0）
    armor.py       戦車の車体・足回り・履帯経路・砲塔・搭載モジュール（0.3.0）
    ship.py        現代艦の船体・上部構造・艤装（0.3.0）
    detail.py      墨入れ・汚し・パネルライン・番号などテクスチャの書き込み（0.3.0）
    io_obj.py      OBJ 読み書き
    io_gltf.py     glb / glTF 読み書き
    cli.py         コマンドライン
    mcp_server.py  MCPサーバー
    AI_GUIDE.md    AI向け手引き
  examples/light_tank.py
  examples/ww2_projects/   WW2期の計画機6機（制作テスト。build_all.py で out/ に書き出す）
  examples/modern/         現代兵器用キットの例4本（単発・双発戦闘機、西側主力戦車、イージス駆逐艦）
  tests/selftest.py, tests/compare_blender.py
  verify/                  動作確認キット（verify.py、確認用モデル、比較用画像、Blender での確認）
```

## 更新履歴

- 0.3.0（2026-10-06）：現代兵器用キットを追加しました。現代パック（vehicleaddonmodern）の制作で見つかった、WW2機向け手法の弱点への対策です。既存の関数の動作は変えていません（動作確認キットの134項目は完全一致のまま）。
  - 追加したモジュール：`planform`（直線の台形翼）、`fuselage`（胴体・ノズル・コックピット）、`gear`（引き込み脚）、`armor`（戦車）、`ship`（現代艦）、`detail`（テクスチャの書き込み）
  - `check` に追加した関数：`penetration`（めり込み）、`compare_size`（公表寸法との照合）、`marking_spot`（標識を描く場所の平らさ）
  - 例：`examples/modern/`（単発戦闘機・双発戦闘機・西側主力戦車・イージス駆逐艦）
  - `tests/selftest.py` に新キットの検査を追加。`AI_GUIDE.md` に6章「現代兵器用キット」を追加
- 0.2.0：動作確認キット `verify/` を追加しました。`check.clearance()`（動く部品の当たり判定）を追加しました。`attach.outside()` に `skip_tags`（脚扉など外板に沿わせる部品を除く）と `samples`（三角形の面上の点も調べる）を追加しました。
- 0.1.0：最初の版です。

