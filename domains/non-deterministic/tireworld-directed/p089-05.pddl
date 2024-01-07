(define (problem tireworld-089-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l9 - location)
    (:init (road l0 l68) (road l1 l85) (road l10 l11) (road l10 l66) (road l11 l56) (road l12 l42) (road l13 l37) (road l14 l42) (road l14 l83) (road l15 l34) (road l15 l46) (road l16 l46) (road l16 l86) (road l17 l47) (road l18 l28) (road l18 l36) (road l19 l54) (road l19 l86) (road l2 l43) (road l2 l87) (road l20 l69) (road l21 l50) (road l21 l87) (road l22 l12) (road l22 l67) (road l23 l59) (road l23 l68) (road l24 l78) (road l25 l33) (road l25 l63) (road l26 l69) (road l26 l79) (road l27 l77) (road l28 l18) (road l29 l31) (road l3 l61) (road l30 l53) (road l31 l20) (road l32 l36) (road l32 l57) (road l33 l71) (road l34 l35) (road l35 l84) (road l36 l18) (road l36 l32) (road l37 l64) (road l38 l72) (road l38 l75) (road l39 l55) (road l39 l6) (road l39 l7) (road l4 l5) (road l4 l71) (road l40 l72) (road l41 l17) (road l41 l74) (road l42 l12) (road l42 l28) (road l43 l2) (road l43 l63) (road l44 l24) (road l44 l85) (road l45 l8) (road l46 l15) (road l47 l17) (road l47 l66) (road l48 l6) (road l49 l5) (road l49 l82) (road l5 l49) (road l50 l21) (road l50 l64) (road l51 l48) (road l51 l70) (road l51 l83) (road l52 l55) (road l53 l30) (road l53 l74) (road l54 l19) (road l54 l81) (road l55 l39) (road l55 l52) (road l56 l60) (road l57 l27) (road l57 l32) (road l58 l30) (road l59 l9) (road l6 l39) (road l6 l48) (road l60 l13) (road l61 l1) (road l61 l3) (road l62 l73) (road l63 l25) (road l63 l43) (road l64 l50) (road l65 l83) (road l66 l10) (road l66 l47) (road l67 l22) (road l67 l51) (road l68 l23) (road l69 l20) (road l69 l26) (road l7 l0) (road l7 l39) (road l70 l45) (road l71 l33) (road l71 l4) (road l72 l38) (road l73 l29) (road l74 l41) (road l75 l88) (road l76 l40) (road l76 l84) (road l77 l27) (road l77 l52) (road l78 l24) (road l78 l80) (road l79 l26) (road l79 l58) (road l8 l22) (road l8 l45) (road l80 l81) (road l81 l54) (road l81 l80) (road l82 l3) (road l82 l49) (road l83 l14) (road l83 l51) (road l83 l65) (road l84 l35) (road l84 l76) (road l85 l1) (road l85 l44) (road l86 l16) (road l87 l2) (road l87 l21) (road l88 l65) (road l88 l75) (road l9 l59) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l88) (vehicle-at l62))
    (:goal (vehicle-at l9))
)