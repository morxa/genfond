(define (problem blocks-013-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b3) (clear b6) (clear b7) (handempty) (on b0 b11) (on b10 b12) (on b11 b4) (on b12 b1) (on b4 b9) (on b5 b10) (on b6 b8) (on b8 b5) (on b9 b2) (ontable b1) (ontable b2) (ontable b3) (ontable b7))
    (:goal (clear b5))
)