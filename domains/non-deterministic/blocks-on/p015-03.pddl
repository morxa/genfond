(define (problem blocks-015-03)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b2) (clear b3) (clear b8) (handempty) (on b0 b11) (on b1 b0) (on b11 b12) (on b12 b5) (on b13 b9) (on b14 b10) (on b2 b6) (on b5 b7) (on b6 b1) (on b7 b14) (on b8 b13) (on b9 b4) (ontable b10) (ontable b3) (ontable b4))
    (:goal (on b8 b14))
)