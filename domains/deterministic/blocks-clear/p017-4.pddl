(define (problem blocks-017-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b16) (clear b5) (handempty) (on b1 b6) (on b10 b11) (on b11 b1) (on b12 b3) (on b13 b9) (on b14 b4) (on b15 b12) (on b16 b2) (on b2 b15) (on b3 b7) (on b4 b13) (on b5 b14) (on b6 b8) (on b9 b10) (ontable b0) (ontable b7) (ontable b8))
    (:goal (clear b10))
)