(define (problem blocks-014-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b10) (clear b5) (clear b6) (handempty) (on b0 b1) (on b1 b7) (on b12 b0) (on b13 b4) (on b2 b9) (on b3 b13) (on b4 b12) (on b5 b8) (on b7 b2) (on b8 b3) (on b9 b11) (ontable b10) (ontable b11) (ontable b6))
    (:goal (clear b2))
)