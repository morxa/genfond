(define (problem blocks-019-08)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b13) (clear b16) (handempty) (on b1 b10) (on b10 b2) (on b11 b6) (on b12 b7) (on b13 b12) (on b15 b8) (on b17 b18) (on b18 b15) (on b2 b14) (on b3 b17) (on b4 b3) (on b5 b9) (on b6 b4) (on b7 b11) (on b8 b5) (on b9 b1) (ontable b0) (ontable b14) (ontable b16))
    (:goal (on b8 b17))
)