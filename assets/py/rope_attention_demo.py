from manim import *
import numpy as np
from manim.mobject.geometry.tips import StealthTip

class RoPEAttentionAnimation(Scene):
    def construct(self):
        # ------------------- 1. 坐标轴配置与初始化 -------------------
        axis_style = {
            "stroke_width": 2,
            "color": WHITE,
            "tip_shape": StealthTip,
            "tip_height": 0.1,
            "tip_width": 0.08,
            "include_tip": True,
        }

        # 构建左、中、右三个独立坐标轴
        ax1 = Axes(x_range=[0, 2.8, 1], y_range=[0, 2.8, 1], x_length=3.2, y_length=3.2, axis_config=axis_style).shift(LEFT * 4.5 + DOWN * 0.3)
        ax2 = Axes(x_range=[0, 2.8, 1], y_range=[0, 2.8, 1], x_length=3.2, y_length=3.2, axis_config=axis_style).shift(DOWN * 0.3)
        ax3 = Axes(x_range=[0, 2.8, 1], y_range=[0, 2.8, 1], x_length=3.2, y_length=3.2, axis_config=axis_style).shift(RIGHT * 4.5 + DOWN * 0.3)

        # 配色定义
        COLOR_BLUE = BLUE_D       # 代表“牛奶”
        COLOR_ORANGE = ORANGE     # 代表“饼干”

        # ------------------- 2. 文本标题构建与染色 -------------------
        # 图 1 标题：未标注位置
        t1 = Text("买了一盒牛奶饼干", font_size=28).next_to(ax1, UP, buff=0.25)

        # 图 2 标题：牛奶 (Pos 4-5) 为蓝色，饼干 (Pos 6-7) 为橙色
        t2 = Text("买了一盒牛奶饼干", font_size=28).next_to(ax2, UP, buff=0.25)
        t2[4:6].set_color(COLOR_BLUE)
        t2[6:8].set_color(COLOR_ORANGE)

        # 图 3 标题：饼干 (Pos 4-5) 为橙色，牛奶 (Pos 6-7) 为蓝色
        t3 = Text("买了一盒饼干牛奶", font_size=28).next_to(ax3, UP, buff=0.25)
        t3[4:6].set_color(COLOR_ORANGE)
        t3[6:8].set_color(COLOR_BLUE)

        # ------------------- 3. 向量数据与计算函数 -------------------
        v_q = np.array([2.0, 1.2, 0.0])
        v_k = np.array([0.8, 1.8, 0.0])
        orig_score = np.dot(v_q, v_k)

        # 计算二维向量分别旋转指定角度后的点乘分数
        def calc_rotated_score(vec_q, vec_k, deg_q, deg_k):
            rad_q, rad_k = np.radians(deg_q), np.radians(deg_k)
            
            # 构建二维旋转矩阵
            R_q = np.array([[np.cos(rad_q), -np.sin(rad_q)], [np.sin(rad_q), np.cos(rad_q)]])
            R_k = np.array([[np.cos(rad_k), -np.sin(rad_k)], [np.sin(rad_k), np.cos(rad_k)]])
            
            q_rot = R_q @ vec_q[:2]
            k_rot = R_k @ vec_k[:2]
            
            return np.dot(q_rot, k_rot)

        # 辅助函数：构建带有动态响应标签的向量箭头
        def create_vector_and_upright_label(ax, vec_coords, label_str, color):
            arr = Arrow(ax.c2p(0, 0), ax.c2p(*vec_coords[:2]), color=color, buff=0, stroke_width=3)
            dir_vec = vec_coords / np.linalg.norm(vec_coords) if np.linalg.norm(vec_coords) != 0 else RIGHT
            
            # 使用 always_redraw 保证箭头旋转时标签位置动态跟随
            label = always_redraw(
                lambda: MathTex(label_str, font_size=32, color=color).next_to(
                    arr.get_end(),
                    direction=dir_vec,
                    buff=0.12
                )
            )
            return arr, label

        # 生成三组坐标轴上的向量与标签
        a1_q, l1_q = create_vector_and_upright_label(ax1, v_q, "q", BLUE)
        a1_k, l1_k = create_vector_and_upright_label(ax1, v_k, "k", RED)

        a2_q, l2_q = create_vector_and_upright_label(ax2, v_q, "q'", BLUE)
        a2_k, l2_k = create_vector_and_upright_label(ax2, v_k, "k'", RED)

        a3_q, l3_q = create_vector_and_upright_label(ax3, v_q, "q''", BLUE)
        a3_k, l3_k = create_vector_and_upright_label(ax3, v_k, "k''", RED)

        # ------------------- 4. 动画播放过程 -------------------
        
        # 步骤 0：初始化所有坐标轴与标题
        self.play(
            Create(ax1), Create(ax2), Create(ax3),
            Write(t1), Write(t2), Write(t3),
            run_time=1.8
        )
        self.wait(0.3)

        # 步骤 1：图 1 展示原始无旋转向量并计算基础分数
        self.play(GrowArrow(a1_q), FadeIn(l1_q), run_time=1.4)
        self.play(GrowArrow(a1_k), FadeIn(l1_k), run_time=1.4)
        s1 = MathTex(f"q \\cdot k = {orig_score:.2f}", font_size=30).next_to(ax1, DOWN, buff=0.3)
        self.play(Write(s1), run_time=1.4)
        self.wait(1)

        # 步骤 2：图 2 展示“牛奶饼干”搭配（q' 旋转 24°，k' 旋转 32°）
        self.play(
            GrowArrow(a2_q), FadeIn(l2_q),
            GrowArrow(a2_k), FadeIn(l2_k),
            run_time=1.2
        )
        self.wait(0.2)

        self.play(Rotate(a2_q, angle=24*DEGREES, about_point=ax2.c2p(0,0)), run_time=1.6)
        self.play(Rotate(a2_k, angle=32*DEGREES, about_point=ax2.c2p(0,0)), run_time=1.6)

        score_fig2 = calc_rotated_score(v_q, v_k, 24, 32)
        s2_updated = MathTex(f"q' \\cdot k' = {score_fig2:.2f}", font_size=30).next_to(ax2, DOWN, buff=0.3)
        self.play(Write(s2_updated), run_time=1.4)
        self.wait(1)

        # 步骤 3：图 3 展示“饼干牛奶”搭配（q'' 旋转 32°，k'' 旋转 24°）
        self.play(
            GrowArrow(a3_q), FadeIn(l3_q),
            GrowArrow(a3_k), FadeIn(l3_k),
            run_time=1.2
        )
        self.wait(0.2)

        self.play(Rotate(a3_q, angle=32*DEGREES, about_point=ax3.c2p(0,0)), run_time=1.6)
        self.play(Rotate(a3_k, angle=24*DEGREES, about_point=ax3.c2p(0,0)), run_time=1.6)

        score_fig3 = calc_rotated_score(v_q, v_k, 32, 24)
        s3_updated = MathTex(f"q'' \\cdot k'' = {score_fig3:.2f}", font_size=30).next_to(ax3, DOWN, buff=0.3)
        self.play(Write(s3_updated), run_time=1.4)
        self.wait(1.5)