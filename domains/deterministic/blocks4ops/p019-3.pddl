(define (problem blocks-019-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b11) (clear b5) (handempty) (on b0 b8) (on b1 b12) (on b10 b9) (on b11 b7) (on b12 b17) (on b13 b2) (on b14 b1) (on b15 b16) (on b16 b10) (on b17 b15) (on b18 b3) (on b3 b4) (on b5 b14) (on b6 b18) (on b8 b13) (on b9 b6) (ontable b2) (ontable b4) (ontable b7))
    (:goal (and (on b5 b7) (on b7 b11) (on b11 b18) (on b18 b4) (on b4 b13) (on b13 b9) (on b9 b2) (on b2 b17) (on b17 b1)))
)