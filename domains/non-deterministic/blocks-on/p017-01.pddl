(define (problem blocks-017-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b14) (handempty) (on b0 b11) (on b1 b16) (on b11 b8) (on b12 b1) (on b13 b10) (on b14 b2) (on b15 b3) (on b16 b6) (on b2 b15) (on b3 b12) (on b5 b4) (on b6 b7) (on b7 b13) (on b8 b9) (on b9 b5) (ontable b10) (ontable b4))
    (:goal (on b13 b6))
)