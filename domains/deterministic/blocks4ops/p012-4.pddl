(define (problem blocks-012-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b4) (handempty) (on b0 b3) (on b1 b6) (on b10 b9) (on b11 b5) (on b2 b7) (on b3 b8) (on b4 b2) (on b6 b11) (on b7 b0) (on b8 b10) (ontable b5) (ontable b9))
    (:goal (and (clear b11) (on b11 b10) (on b10 b6) (on b6 b0) (on b0 b5) (on b5 b8) (on b8 b9) (on b9 b4) (on b4 b7) (on b7 b1) (on b1 b2) (on b2 b3) (ontable b3)))
)