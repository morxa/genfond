(define (problem blocks-017-04)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b15) (clear b9) (handempty) (on b0 b1) (on b1 b10) (on b10 b4) (on b12 b0) (on b13 b14) (on b14 b11) (on b15 b2) (on b16 b3) (on b2 b8) (on b3 b13) (on b5 b12) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b16) (ontable b11) (ontable b4))
    (:goal (on b1 b8))
)