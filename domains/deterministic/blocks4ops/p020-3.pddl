(define (problem blocks-020-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b11) (clear b2) (handempty) (on b0 b5) (on b1 b15) (on b10 b14) (on b11 b19) (on b12 b17) (on b13 b6) (on b14 b4) (on b15 b12) (on b16 b13) (on b17 b18) (on b18 b0) (on b19 b10) (on b3 b7) (on b4 b9) (on b5 b16) (on b6 b3) (on b8 b1) (on b9 b8) (ontable b2) (ontable b7))
    (:goal (and (on b16 b19) (on b19 b3) (on b3 b17) (on b17 b15) (on b15 b1) (on b1 b8) (on b8 b7) (on b7 b12) (on b12 b2) (on b2 b11) (on b11 b13) (on b13 b5) (on b5 b14) (on b14 b18)))
)