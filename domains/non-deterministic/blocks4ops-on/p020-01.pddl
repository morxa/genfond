(define (problem blocks-020-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b12) (clear b2) (clear b7) (handempty) (on b0 b13) (on b1 b11) (on b10 b9) (on b11 b8) (on b12 b6) (on b13 b10) (on b14 b3) (on b16 b0) (on b18 b16) (on b19 b18) (on b2 b19) (on b3 b5) (on b4 b17) (on b5 b1) (on b6 b15) (on b7 b14) (on b8 b4) (ontable b15) (ontable b17) (ontable b9))
    (:goal (on b8 b9))
)