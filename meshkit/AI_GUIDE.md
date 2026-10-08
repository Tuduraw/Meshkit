# meshkit 作業ガイド（AI向け）

Blenderを使わずに、Pythonスクリプトでローポリモデルを作り、検査し、プレビュー画像で確かめ、OBJ / glTF に書き出すための手引きです。
MCPの `guide` ツールはこの文書を返します。

## 1. 作業の流れ

1. 寸法と座標系を決める（1単位＝1m、+X＝左、+Y＝上、+Z＝前）。
2. `build()` が `Model` を返すスクリプトを書く。
3. 実行する：MCPなら `run_script`、シェルなら `python -m meshkit run script.py`。
4. 返ってくる報告を読む。`ok` が true になるまで直す。
   - `issues`：穴、巻き順の不一致、裏返り、つぶれた面、平らでない面、凹の面
   - `floating`：どこにも接していないシェルの組（浮き部品）
   - `summary.size`：外形寸法。公表値と比べる
5. プレビュー画像を見る。おかしいところはモードや視点を変えて確かめる。
   - `mode=orientation`：裏面が赤く出る。穴や面の裏返りが一目でわかる
   - `mode=parts` と `wire=true`：パーツの分かれ方とポリゴンの割り方がわかる
   - `views=left,top,front`：正投影。図面と見比べられる
6. 数値を直して、3から繰り返す。形を手で直すのではなく、スクリプトを直す。

## 2. 座標系と単位

- メッシュキット内部は +X＝左、+Y＝上、+Z＝前、1単位＝1m（glTF、Minecraft/tudursvehiclemod と同じ）。
- Blenderで作られたOBJ（Z上）を読むときは `axes="blender"`（CLIでは `--axes blender`）を指定する。
- 原点の決め方は用途で決める。航空機は主輪の接地点、戦車は接地面の中心、艦艇は喫水線など。
- 可動部（砲塔・舵・脚など）は別パーツにし、`model.pivots[パーツ名] = (x, y, z)` に回転の中心を入れる。glTFではこれがノードの原点になる。

## 3. API

```python
import meshkit as mk
import numpy as np

def build():
    m = mk.Model("name")
    m.add("hull", mk.box(0, 0.5, 0, 2, 1, 4, tag="hull"))    # パーツ "hull" にシェルを追加
    m.pivots["$turret"] = (0, 1.0, 0)                          # 可動部の回転中心
    m.refs.append(("seat", (0, 1.2, 0), "seat"))               # 参照点（seat / muzzle / camera / 任意）
    return m
```

### 形状（どれも閉じた外向きのシェルを返す）

| 関数 | 用途 |
|---|---|
| `box(cx, cy, cz, sx, sy, sz, tag, M=None, pivot=None)` | 箱。`M` は回転行列（`rot_x/rot_y/rot_z/rot_axis`） |
| `cylinder(p0, p1, r0, r1=None, n=8, tag)` | 円柱・円錐台（任意の軸） |
| `lathe(p0, axis_dir, [(t, r), ...], n=8, tag)` | 回転体。r=0 の端は尖る（砲弾・スピナー・煙突） |
| `loft([ring, ring, ...], tag, tags=None)` | 断面の列をつなぐ。断面は同じ点数。1点の断面は尖端 |
| `ring_superellipse(cx, cy, z, hw, ht, hb=None, n=12, p=2.0)` | z一定の面上の断面（p=2楕円、p>2で角ばる）。胴体・船体・砲塔のロフト用 |
| `ring_on_axis(center, axis, radius, n)` | 任意軸まわりの円断面 |
| `prism(bottom, top, tag)` | 同じ点数の2つの多角形の間の立体 |
| `extrude_xz(poly, y0, y1)` | 平面図（x, z）の多角形を上下に押し出す（甲板構造物） |
| `extrude_zy(poly, x0, x1)` | 側面図（z, y）の多角形を左右に押し出す（車体側面・舵） |
| `extrude_xy(poly, z0, z1)` | 正面図（x, y）の多角形を前後に押し出す |
| `ellipsoid(center, rx, ry, rz, nu, nv)` | 楕円体 |
| `wing(stations)` / `fin(stations)` / `wing_section(...)` / `airfoil_2d(t, camber)` | 翼（NACA断面、ねじり、翼端の丸め） |
| `blade(root, tip, chord_root, chord_tip, thick, axis_spin)` | プロペラ羽根 |

シェルの操作：`.translate(x, y, z)`、`.rotate(M, pivot)`、`.apply(M, t, pivot)`、`.mirrored_x()`（左右反転コピー、巻き順も直す）、`.copy()`、`.retag(tag)`。

### テクスチャ

```python
from meshkit.texture import texture_model, value_noise
texture_model(m, (512, 512),
              colors={"hull": (100, 110, 70)},                 # 投影する面の地色
              flat={"track": (50, 50, 46), "gun": (60, 64, 52)},  # 単色の色見本に割り当てる小物
              paint=lambda atlas, model: atlas.paint3d(fn, tags=["hull"]))
```

- 面は法線の向きで上・下・側面・正面のどれかの投影図に割り当てられる。塗装は3D座標の関数で書く（`atlas.paint3d(fn(X, Y, Z, cur) -> (rgb, mask))`）。
- 左右で違う文字や番号を塗るときは `split_sides=True`。ガラスなど透ける面は `glass_tags` と `atlas.paint_alpha`。
- 実行すると `model.uvs` と `model.texture` が入る。色分けだけでよければ `flat="all"`。

### 検査・表示・入出力

```python
from meshkit import check, view
rep = check.report(m)                          # ok / issues / floating / summary
view.sheet(m, views=("iso", "left", "top"), mode="orientation", wire=True).save("p.png")
view.render_view(m, {"eye": (8, 4, 8), "target": (0, 1, 0), "fov": 30})  # 任意のカメラ
check.clearance(m, "$propeller", m.pivots["$propeller"], (0, 0, 1),   # 回す部品が他に当たらないか
                tags=("prop",))                                     # （羽根だけ・1周）
mk.save(m, "out.obj"); mk.save(m, "out.glb")
m2 = mk.load("other.glb")                      # OBJ / GLB / glTF を Model として読む
```

- 視点：`iso`、`iso_rear`、`iso_below`、`iso_left`、`iso_right`（透視）、`front`、`back`、`left`、`right`、`top`、`bottom`（正投影、自動で収める）。
- `left` は +X 側（モデルの左舷）から見た図で、機首が画像の左に来る。`top` は機首が上。
- 参照点（`model.refs`）の印は隠れていても手前に描かれる（seat＝黄、muzzle＝赤、camera＝水色）。
- 回転・折りたたみ・旋回する部品は `check.clearance(model, part, pivot, axis, angles, tags, ignore)` で、動かしたときに他の部品へ当たらないかを確かめる（例：プロペラの羽根を1周、折りたたみ翼を0〜110度）。
- 引き込む部品（脚・車輪）は `attach.outside(model, transforms, parts, skip_tags=("door",), samples=3)` で、引き込んだ姿勢で機体の外にはみ出していないかを確かめる（外に出ている表面点の割合。0が目標。`samples` は三角形の面上の点も調べる細かさで、頂点の間で外板を突き抜けるのも見つかる）。脚扉のように外板に沿わせる部品はタグで除く。
- 浮いた部品の検出は `attach.groups(model, transforms)`（つながったシェルのまとまり。2つ以上なら浮きがある）。
- 姿勢を変えて見る：`view.render_view(m, "iso", transforms={"$gear_l": fn})`。fn は点の配列を受け取って返す関数。

## 4. 守ること

- 各シェルは閉じたメッシュにする。どの辺もちょうど2面が逆向きに共有する。開いた形が欲しいときも、薄い板（厚み数mm〜数cm）にする。
- 部品は少し食い込ませてつなぐ（数cm）。同じパーツの2つのシェルで頂点の位置を完全に一致させない。
- 多角形の面は平らで凸にする。凹の断面は三角形に分けるか、`loft` の端に任せる（凹なら自動で扇状に分割される）。
- 背面カリング（裏面を描かない）で表示される前提で作る。`mode=orientation` で赤い部分が出たら、穴か裏返り。
- ガラスの内側には内装（床・座席・計器盤など）を入れる。外板の裏は描かれないので、空洞に見える。
- 三角形数を報告で確かめる（目安：戦闘機・戦車は約1,000、大型機は2,000以下、艦艇は10,000以下）。

## 5. コマンドライン

```
python -m meshkit run script.py -o out --views iso,left,top --mode parts --wire
python -m meshkit render model.obj --texture tex.png --views iso,iso_rear --mode texture
python -m meshkit render blender_export.obj --axes blender --mode orientation
python -m meshkit check model.glb
python -m meshkit info model.obj
python -m meshkit convert model.obj model.glb --texture tex.png
python -m meshkit diff old.obj new.obj
python -m meshkit silhouette model.obj --view side -o side.png --ref drawing.png --ref-box -4.6,-0.2,4.6,3.0
```

## 6. 現代兵器用キット（0.3.0）

現代パック（ジェット機・主力戦車・現代艦艇42種）の制作で、WW2機向けの手法では形が実物から外れる点がいくつも見つかりました。
その対策をモジュールにまとめたものです。例は `examples/modern/` の4本（単発戦闘機・双発戦闘機・西側主力戦車・イージス駆逐艦）です。

### 6.1 モジュール一覧

| モジュール | 主な関数 | 用途 |
|---|---|---|
| `meshkit.planform` | `from_published(面積, 翼幅, 前縁後退角, テーパ比, z_le_root, y, dihedral, x_root)`、`trapezoid(半翼幅, 翼根弦, 翼端弦, 後退角, ...)`、`surface(half)`、`fin(root, tip, x, cant, mirror_x)`、`plate(outline)`、`split(half, x)`、`station_at`、`describe(half)` | 直線の台形翼・尾翼・カナード・垂直尾翼・ストレーキ。翼端は丸めず切り落とす |
| `meshkit.fuselage` | `body`、`pod`、`canopy`、`body_with_tub(stations, tubs)`、`cockpit_fittings`、`cover_nozzles(stations, nozzles)`、`nozzle`、`intake_face`、`rider_feet_y` | 胴体・ナセル・テールブーム、コックピットのくぼみ、ノズルを内包する後部胴体 |
| `meshkit.gear` | `leg`、`wheel`、`Retraction(pivot, axis, angle, shift)`、`fit`、`shift_grid`、`fairings`、`stow(model, legs)` | 引き込み脚。角度と平行移動を探索し、残りは脚収納部の膨らみで覆う |
| `meshkit.armor` | `glacis(...)`、`angle_of`、`hull(...)`、`running_gear`、`track_path`、`link_spacing`、`lay_links`、`faceted_turret(..., bustle=)`、`cast_turret(rings)`、`turret_half_width`、`gun`、`sight`、`periscope`、`smoke_bank`、`basket`、`drums`、`searchlight`、`rws`、`bricks_on_plate`、`bricks_grid` | 戦車の車体側面形、足回りと履帯経路、砲塔、搭載モジュール |
| `meshkit.ship` | `hull(stations, flare)`、`deck_at`、`sym`、`house(plan, y0, y1, plan_top / inset)`、`wall_point`、`array_face`、`bridge`、`funnel`、`pyramid_mast`、`lattice_mast`、`yard`、`radome`、`dish`、`whip`、`rail_line`、`deck_edge_rails`、`bollards`、`anchors`、`boat`、`davit`、`raft_rack`、`canisters`、`triple_tubes`、`vls`、`ciws` | 現代艦の船体、傾斜壁の多角形上部構造、艤装 |
| `meshkit.detail` | `ink`、`tone`、`grime`、`streaks`、`chips`、`soot`、`panel_lines`、`rivets`、`access_panels`、`lines_fn`、`rect_fn`、`grid_fn`、`wheel_hubs`、`side_number`、`digit_mask` | テクスチャの書き込み（墨入れ・汚し・パネルライン・番号）。ポリゴンを増やさずに細部を出す |
| `meshkit.check`（追加） | `penetration(model, A, B, tags_b=)`、`compare_size(model, {"length", "width", "height"})`、`marking_spot(model, center, radius, side)` | 部品のめり込み、公表寸法との照合、標識を描く場所が平らか |

### 6.2 ジェット機

- **翼は直線の台形にする。** 前縁・後縁は直線、翼端は切り落とし（`planform.surface`）。`geom.wing` の既定（`round_tip=True`）はレシプロ機向け。
- **翼弦は推定せず公表値から出す。** 翼面積・翼幅・前縁後退角・テーパ比から `from_published` で作り、`describe` で面積・後退角を確かめる。
- **途中の折れは実機にある時だけ。** F-4 や A-10 の外翼のように実機に折れがある場合だけ `trapezoid(..., crank=(x, 後退角, 上反角))` を使う。
- **胴体幅の広い機体・ブーム付きの尾翼は左右別パネル。** ステーションを x > 0 から始めると、`surface` が左右2枚にする。
- **双発機の後部胴体はノズル2基を内包する幅にする。** `cover_nozzles` が後部ステーションを自動で広げる。そのうえで `check.penetration(m, ["tail"], ["fuselage"], tags_b=("nozzle", "exhaust"))` が0であることを確かめる（尾翼がノズルを貫通していない）。
- **国籍標識は平らな面に置く。** `check.marking_spot` で、標識の円が面からはみ出さないか（coverage が1）、段差をまたがないか（relief が小さい）を確かめる。主翼が胴体の最後部まで付いていない機体（デルタ翼など）は特に注意。
- **コックピット**
  - `body_with_tub` でキャノピーの下にくぼみを作り、`cockpit_fittings` でシートと計器盤を置く。
  - 乗員の足元を座席位置に置くエンジンでは、`rider_feet_y(floor)` で腰がシートに乗る高さを出す。tudursvehiclemod は0.7倍の縮尺で描くため、腰は足元から約0.49 m上になる。
- **引き込み脚**
  - `gear.stow` で角度（前脚は前後どちらに畳むかも含む）と、畳んだ後の平行移動を探索する。
  - 残りは `fairings` で左右対称の膨らみを付け、はみ出しを **0%** にする。
  - 「回転してから自分の座標で平行移動」するエンジン（tudursvehiclemod の `slide_rotate`）には `Retraction.rotate_then_slide_offset()` を渡す。

### 6.3 主力戦車

- **車体前面は角度から作る。** `armor.glacis(z_nose, nose_y, roof_y, upper_deg, lower_deg, ...)` を使う。
  - 西側の第3世代：上部前面は水平から約8〜12度（長く緩い）、下部前面は約35〜45度（短く急）。
  - ソ連・ロシア系：上部前面は約22〜30度。
- **西側戦車の車体上面は平らではない。** 緩い前面 → 砲塔リング付近の平坦部 → 機関室で一段上がり後方へ緩く上る（`hull(..., rear_deck=(z_front, z_rear, y_front, y_rear))`）。砲塔の張り出しは `faceted_turret(..., bustle=(z_split, 底上げ量))` で底を上げ、旋回しても機関室に当たらないようにする。
- **車体上部を絞るのは T-54/55 世代だけ。** `hull(..., upper_w=..., fender_y=...)` で履帯の上にフェンダーを付ける。T-72/80/90 と M60 は全幅。
- **差別化は砲塔形状とモジュールで行う。** 鋳造砲塔は `cast_turret` で、前半分の指数 p_front < 2 にすると尖った機首（M60）になる。溶接砲塔は `faceted_turret` で作る。
- **全幅は公表値に合わせる。** 楔形の前面装甲は砲塔の外形そのものに組み込み、外へ張り出す板を足さない（10式で全幅が3.6 mになった失敗）。`check.compare_size` で確かめる。
- **モジュールは面に接して置く。** `turret_half_width(turret, z, 高さの割合)` で砲塔側面の位置を求めて置くと、浮きも外への張り出しも出ない。
- **履帯**
  - `track_path` で経路を作り、`link_spacing` でコマの間隔を出す。
  - エンジンが履帯を描く場合でも、プレビューと接触確認には `lay_links` でコマを並べる。

### 6.4 現代艦艇

- **個性は上部構造で出る。** `house` は傾斜壁の多角形デッキハウス。上面を広くすれば窓が外へ傾いた艦橋になる（`bridge`）。
- **レーダーのアレイ面は壁に沿わせる。** `wall_point` で傾斜壁上の位置を出し、`array_face` を壁の法線方向に向けて置く。
- **三角形の配分**
  - 喫水線より下は、ビルジとキールの点だけで十分（`hull`）。三角形は水線より上に使う。
  - 目安：駆逐艦・巡洋艦は5,000〜7,000、戦艦・空母は5,000〜10,000。
- **艦番号は左右どちらからも読めるようにする。** `texture_model(..., split_sides=True)` と `detail.side_number` を使う。
- **小物は必ず何かに接する。** 艇は甲板や構造物の上に載せ、ヤードはマストと同じ前後幅（`yard(..., depth=)`）にする。手すり・ボラードは `deck_at` で甲板の高さに合わせる。

### 6.5 テクスチャの書き込み

- **順番**：地色・迷彩 → `tone` → パネルライン・リベット・点検パネル（航空機）／外板の継ぎ目・VLSのふた（艦）／転輪のハブ・グリル（戦車） → 標識・番号 → `ink` → `grime`・`streaks`・`chips`・`soot`。
- **`ink`（墨入れ）**：投影面ごとに、面の種類・奥行きの段差・傾きの折れを検出して線とぼかしを入れる。ローポリの箱でも形が読めるようになる。
- **解像度の目安**：航空機・戦車は1024px、艦艇は2048px。
- **注意**：点検パネルや扉は規則的に散らした推定なので、実物の配置を再現したい部分は `rect_fn` で個別に描く。

## 7. ロボット（メカ）用キット（0.4.0）

Mech Frame（tudursvehiclemod のロボットアドオン。標準機9機、変形セット3種、拡張パック4社42パーツ）の制作で得た手法をまとめたものです。
形状の関数は `meshkit.mech`、例は `examples/mech/concept_legs.py`（同じ関節位置の脚を4つのデザイン言語で作り分け、数値を比べる）です。

### 7.1 モジュール（`meshkit.mech`）

| 関数 | 用途 |
|---|---|
| `stack` / `seg` / `cbox` / `cyl` / `mirror` / `nozzle` | 面取り矩形の積み上げ・折れ線ロフト、面取り箱、軸、左右反転、ノズル |
| `pent2d` / `pstack` / `pseg` / `zloft` | 前を向く丸い五角形の断面（軽量機の言語）。正面は細く、側面は長く尖る |
| `plate` / `slab` / `hbox` | ボルト留めの装甲板、凸多角形の板、強く面取りした塊（重装甲の言語） |
| `tube` / `truss` | むき出しの管と格子桁。`truss(..., n=4)` で角材にすると三角形が1/3 |
| `beam` / `girder` / `at` / `panel` / `claw` | 骨抜き機の言語：角材（`beam`）、斜材付きの三角断面の桁（`girder`）、支柱で浮かせた別体の装甲板（`panel`。支柱が板に届かないと例外）、6三角形の爪・棘（`claw`） |
| `chine_seg` / `chine_stack` / `split_warped` | 鋭い稜線を持つ六角形断面（ステルスの多面体言語）。形の変わるロフトのねじれた四角形は自動で三角形に分ける |
| `cast` / `cast_seg` | 超楕円断面（p≒2.6）の鋳造装甲の塊（要塞の言語） |
| `R3` / `axis_angle` / `align` / `frame_rot` / `M4` / `quat` / `slerp` | 関節の回転（変形の姿勢づくり） |
| `volume` / `fill_ratio` / `proportions` / `limb_report` | コンセプトの数値化（充填率、正面幅/側面奥行き）と脚の寸法・休止時の曲がり |

### 7.2 部品の作り方

- **動く単位ごとにパーツ（グループ）を分ける。** 骨盤、腿、すね、足、肩、上腕、前腕、頭。回転の中心（関節）を決めてから形を作る。
- **関節は両側の部品に食い込む軸（円柱）で作る。** 軸は子の側に入れる（膝の軸はすねのグループ）。親の端は軸の外周に数cm食い込ませる。脚の部品が離れて見える原因の多くは、ここが「接しているだけ」になっていること。
- **脚は股・膝・足首の点で決める。** 膝には休止状態で20〜40度の曲がりを付け、曲がる向き（pole）を決める。真っ直ぐだとIKが曲げる向きを決められない（`limb_report` の `bends` で確かめる）。
- **腿：すね≒1：1**（逆関節は短い腿・短い中間節・長いすね）。腿を長くすると、深く曲げたときに膝や第2関節が腿にめり込む。
- 左右対称の部品は左を作って `mirror` する。可動部の名前は左右で `l_` / `r_` をそろえる。

### 7.3 コンセプトからデザインへ

コンセプトを1文で決め、それを「断面」「関節の扱い」「塊の置き方」「色の役割」の4つの規則に落とします。規則が決まれば、どの部品もその会社の物に見えます。

| 言語 | コンセプト | 断面 | 関節 | 塊・シルエット | 充填率の目安 |
|---|---|---|---|---|---|
| 軽量 | 速さのために細く | 前を向く丸い五角形 | 細い首・腰・足首、肩・肘・膝だけ太く | 正面は細く側面は深い（正面幅/側面奥行き < 0.8） | 0.3前後 |
| 骨抜き（超軽量） | 飛ばない物はすべて重り | 斜材で三角形に組んだ角材の桁。装甲は別体の尖った薄板を支柱で浮かせて並べる。操縦席と感覚器だけ卵形の殻 | 円盤だけの軸、アクチュエーター（ラム・ばね）を見せる | 背が高く細い、板の先端が上・前・後ろに尖る、過大なスラスター | 0.05〜0.1 |
| ステルス | 最後に見つかる | 稜線のある六角形、平面だけ | 六角形の軸、くぼんだスリット | 直線だけの輪郭、縁の角度をそろえる、鋸歯の縁 | 0.3前後 |
| 要塞（超重量） | 歩く城 | 鋳造の超楕円 | 何重ものスカート板と膝当てで覆う | 肩が脚より広い、頭は肩より低く埋まる | 0.45以上 |

- **骨抜きでも、自重を支えられる構造に見せる。** 細い管を1本つないだだけの手足は「歩くだけで折れそう」に見える（指摘を受けた点）。部材は角材にして弦材を2〜3本並べ、斜材で三角形に組む（`girder`）。胴は輪と縦通材の箱に、側面へX字の筋交いを入れる。
- **骨抜き機の装甲は「骨組みに沿って浮かせた別体の板」。** 板は骨組みの面と平行に、面から 0.1 m ほど離して支柱で留め、板どうしも 0.05〜0.1 m 空ける（関節や肩の張り出しの所で分割する）。隙間から骨組みが見えるのが軽さの表現。板は覆うためでなく空気を受け流すために置く：進行方向（+Z）に対して斜めか水平にし、正面を向く板は作らない。胴の前は骨組みの縦材を1点に集めた円錐の機首にして、その面ごとに板を張る。手足の板は外側面に縦に並べ、膝は脚の面内のひれにする。板の外形は尖った五角形で、先端を上・前・後ろへ向けて鋭いシルエットにする。
- **重量級ほど関節を太く・隠す。軽量級ほど関節を細く・見せる。** 同じ骨組みでも、関節の扱いだけで重さの印象が変わる。
- **色は7つの役割で塗る**：armor（地）、armor2（第2色）、frame（関節・内部）、accent（差し色）、eye（発光）、nozzle、metal（＋glass）。会社ごとにこの組を1つ決め、差し色は「アクチュエーター」「縁」「先端」など置く場所の規則も決める。暗い配色は面の陰影が見えにくいので、armor と armor2 の明度差を大きめに取る。
- **変形機は同じ会社の通常機の部品関数を使い回し**、翼パックや機首など変形に必要な物だけ足す。系統が一目で分かり、作業も減る。

### 7.4 脚の種類別

- **二脚**：股の高さ 2.9〜3.3 m。足は前に長い五角形（または六角形）の塊で、つま先は足首より 0.6〜0.9 m 前。
- **逆関節**：横から見て稲妻形。短い腿が前下へ、短い中間節が後上へ（第2関節は膝より高い）、長いすねが前下へ。
- **多脚**：胴の縁に付け根の回転ブロック（coxa、縦軸）、腿は外上へ上がって胴より高い膝、すねは外下へ。関節は上と前から板（フェンダー）で守る。四脚は対角、六脚は三脚ずつの組で歩くので、隣の脚と前後の可動範囲が重ならない間隔にする。
- **タンク**：車体はどの重量級でも閉じた水密の箱（船形）にする。骨抜き機でも格子だけの車体にはしない。履帯は台形（短い接地面、前の誘導輪へ上がる斜面、高い起動輪、長い上部）。転輪は回転グループ（中心と半径）。スカートで上部を隠し、転輪は下から見せる。
- **ホバー**：脚なし。スカートは接地面から 0.3〜0.5 m 浮かせ、下面に発光、周囲にファンのダクト。

### 7.5 変形

- **ジェット形態の配置は手で打たない。** 各グループを親に関節でぶら下げ、ロボット形態の配置を関節まわりに回したものを使う（Mech Frame の `rig.Rig`）。こうすると両形態でも変形の途中でも、関節が外れない。
- **確かめるのは t=0.25 / 0.5 / 0.75 / 1.0 の4コマすべての浮き部品**。最終形だけ見ても途中の外れは分からない。
- よく使った形：
  - コアの寝かせ方：`R3(rx=-90, ry=180)`（頭が前・胸が上）、`R3(rx=-90)`（頭が後ろ・胸が上）、`R3(rx=90)`（頭が前・胸が下）。
  - 脚：まっすぐ後ろへ伸ばして閉じる（股を内へずらす `slide`）＝尾部や機首。すねを前へ折り返す（Z折り）＝全長を詰める。すねを軸まわりに転がすと、ふくらはぎのフィンがV尾翼や水平尾翼になる。
  - 翼：背中に垂らして畳み、関節で横へ振り出す（`frame_rot` で翼幅・翼弦・面の向きを一度に決める）。外側の板は内側に重ねて畳み、ヒンジで180度開く。ヒンジが翼弦方向なら翼幅が倍、翼幅方向なら翼弦が倍になる。
- 噴射口・当たり判定の高さは、リグで変換した点から計算して書き出す（手で測らない）。

### 7.6 三角形の予算

- 目安は**武器込みで1機5,000以下**。武器が4つで約900使うので、脚1,400〜1,800、コア700〜1,000、腕600〜1,000、頭200〜350、ブースター500〜650。
- 多面体（六角形断面）は安く、管と格子は高い。格子は角材（n=4）、管は継ぎ目のリングを省く（`tube(..., joint=0)`）、リングは4面の角柱を並べる、小さな発光窓は板1枚にする。

### 7.7 動きを見越した形

歩行の調整で分かった、形の側で決めておくべきこと。

- **足の前後の幅は歩幅の半分以下に。** 大きな足は踏み出しで隣の足や反対の脚に当たる。
- **多脚は脚ごとに前後の「通り道」を持てる間隔で付け根を並べる。** 付け根の間隔が狭いと、どんな歩き方でも隣どうしが重なる。
- **膝の休止時の曲がりが、着地の沈み込みとジャンプの伸びの余裕になる。** 伸び切った脚は着地を吸収できない。
- 二脚は脚の外側に揺れる物（スカート、フィン）を付けすぎない。横移動のとき脚が外へ開く。

### 7.8 検査

- 部品ごとに `check.report`（穴・裏返り・凹の面）、組み上げた機体でグループ単位の浮き部品、変形機は途中の4コマ。
- プレビューは `mode=tags` で会社の配色を当てて、斜め・横・正面・上の4方向。シルエットは横と正面で判断する。
- `proportions` / `fill_ratio` で会社ごとの数値がそろっているか見る。骨抜き機に塊が混ざると充填率が跳ね上がる。
