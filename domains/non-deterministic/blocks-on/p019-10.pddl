(define (problem blocks-019-10)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b10) (clear b12) (clear b17) (handempty) (on b0 b1) (on b1 b11) (on b10 b9) (on b11 b3) (on b13 b5) (on b14 b2) (on b15 b4) (on b16 b0) (on b17 b14) (on b18 b13) (on b2 b8) (on b3 b18) (on b5 b6) (on b7 b16) (on b8 b15) (on b9 b7) (ontable b12) (ontable b4) (ontable b6))
    (:goal (on b15 b11))
)