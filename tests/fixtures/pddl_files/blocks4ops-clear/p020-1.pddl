(define (problem blocks-020-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b19) (clear b3) (clear b8) (handempty) (on b0 b7) (on b10 b13) (on b11 b17) (on b12 b11) (on b13 b4) (on b14 b2) (on b15 b0) (on b16 b18) (on b17 b9) (on b18 b10) (on b2 b16) (on b3 b6) (on b4 b12) (on b5 b14) (on b6 b5) (on b9 b15) (ontable b1) (ontable b19) (ontable b7) (ontable b8))
    (:goal (clear b15))
)