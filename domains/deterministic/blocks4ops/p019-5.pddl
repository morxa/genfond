(define (problem blocks-019-5)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b11) (clear b8) (handempty) (on b0 b5) (on b1 b17) (on b10 b14) (on b11 b12) (on b12 b4) (on b13 b0) (on b14 b1) (on b16 b3) (on b18 b15) (on b2 b13) (on b3 b18) (on b4 b7) (on b5 b10) (on b6 b9) (on b7 b16) (on b8 b6) (on b9 b2) (ontable b15) (ontable b17))
    (:goal (and (on b11 b13) (on b13 b8) (on b8 b18) (on b18 b6) (on b6 b12) (on b12 b17) (on b17 b15) (on b15 b4) (on b4 b1) (on b1 b7) (on b7 b10) (on b10 b0)))
)