(define (problem blocks-016-03)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b11) (clear b14) (clear b5) (handempty) (on b0 b2) (on b1 b9) (on b10 b6) (on b12 b3) (on b13 b7) (on b15 b4) (on b2 b1) (on b3 b13) (on b4 b8) (on b6 b15) (on b7 b10) (on b9 b12) (ontable b11) (ontable b14) (ontable b5) (ontable b8))
    (:goal (on b12 b7))
)