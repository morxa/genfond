(define (problem blocks-011-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b10) (clear b3) (clear b9) (handempty) (on b0 b7) (on b10 b8) (on b2 b6) (on b4 b1) (on b5 b2) (on b7 b5) (on b9 b4) (ontable b1) (ontable b3) (ontable b6) (ontable b8))
    (:goal (clear b7))
)