(define (problem blocks-020-09)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b10) (clear b13) (handempty) (on b0 b1) (on b1 b14) (on b10 b19) (on b11 b6) (on b13 b5) (on b14 b7) (on b15 b18) (on b16 b15) (on b18 b9) (on b19 b2) (on b2 b11) (on b3 b4) (on b4 b8) (on b5 b12) (on b6 b3) (on b7 b17) (on b8 b16) (on b9 b0) (ontable b12) (ontable b17))
    (:goal (on b19 b15))
)